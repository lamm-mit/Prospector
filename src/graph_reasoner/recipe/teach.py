"""Stage 0 of the recipe: the teacher pass that produces the demonstrations
stage 1 trains on and stage 3 blends, run on your own machine with the API
key in `.env`.

    prospector teach --graph tsmc 0                 # shard 0 first: its "LLM usage" line prices the other seven
    prospector teach --graph tsmc 1 2 3 4 5 6 7     # the rest, in sequence; a shard whose summary exists is skipped
    prospector teach --graph tsmc ids               # pass-1 rows and the failed-id list for the second draw
    prospector teach --graph tsmc p2                # the second draw, over the failed ids only, uncached
    prospector teach --graph tsmc mix               # rows from all nine runs, detours, the mix and its sidecar

The teacher (`--teacher`) answers the graph's
training questions with the three tools, 20 turns, 3,000 tokens a turn, the
think-note at generation time (stripped from the rows by `build_sft_rows.py
--strip-user-note`), in tier-balanced shards; one more draw over the items
pass 1 failed gives each item at most one verified episode; then the rows
(cap 1, gated_acc, the evaluation questions excluded), the synthetic detours
and the two-part mix. `--tag` names the pass and every file it writes
(`graph_reasoner.recipe`), so a tag holds one teacher's pass. The teacher is
the one the tag's stored runs and rows record, or google/gemini-3.7-flash for
a new tag; a `--teacher` that contradicts the record is refused, and so is
any teacher but google/gemini-3.7-flash under the default tag, which is named
after it.
Another teacher takes another tag. The client routes by model id: a vendor-prefixed
id goes through OpenRouter, a bare OpenAI id runs on `OPENAI_API_KEY`; the
keys come from `.env` and never go on a command line.
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

from ..benchmarks import run_dir_name
from .names import DEFAULT_TAG, Layout
from .run import Failed, Refused, Runner, log_lines, say, shown

DEFAULT_TEACHER = "google/gemini-3.7-flash"
NOTE = ("Before every tool call, first write one or two sentences inside <think>...</think> explaining why "
        "this call is the right next step given what you have observed so far. Use the same <think> block "
        "before your final answer.")


class Teach:
    def __init__(self, layout: Layout, runner: Runner, *, teacher: str | None = None, shards: int = 8):
        self.L, self.runner, self.teacher, self.shards = layout, runner, teacher, shards
        self.bench = layout.graph.train_benchmark
        self.core = layout.graph.eval_benchmark
        self.name = layout.teach_run_prefix
        (layout.root / "logs").mkdir(exist_ok=True)
        (layout.root / "data" / "traj").mkdir(parents=True, exist_ok=True)

    def run_dir(self, run: str) -> str:
        return f"data/runs/{run_dir_name(run, self.bench)}"

    def shard_runs(self) -> list[str]:
        return [f"{self.name}-g{g}" for g in range(self.shards)]

    def have(self, rel: str) -> bool:
        return (self.L.root / rel).exists()

    def recorded_teachers(self) -> dict[str, str]:
        """The model each stored record under this tag names, by where it
        is: a teacher run's summary, or the rows' sidecar. None for a new tag."""
        L, found = self.L, {}
        for summary in sorted((L.root / "data" / "runs").glob(f"{run_dir_name(self.name + '-*', self.bench)}/summary.json")):
            if model := json.loads(summary.read_text()).get("model"):
                found[str(summary.parent.relative_to(L.root))] = model
        for rows in (L.demos_keeps_pass1, L.demos_keeps):
            sidecar = L.root / f"{rows}.provenance.json"
            if sidecar.exists() and (teacher := json.loads(sidecar.read_text()).get("teacher", "unknown")) != "unknown":
                found[rows] = teacher
        return found

    def check_teacher(self) -> None:
        """Settle the teacher: the one asked for, else the one the tag's stored
        pass records, else DEFAULT_TEACHER. One that differs from the record is
        refused: its shards would be skipped as done, and its rows built from
        the recorded teacher's runs. So is any teacher but DEFAULT_TEACHER
        under DEFAULT_TAG, whose name would then mislabel every file."""
        recorded = self.recorded_teachers()
        teacher = self.teacher or next(iter(recorded.values()), DEFAULT_TEACHER)
        for where, model in recorded.items():
            if model != teacher:
                raise Refused(f"the tag {self.L.spec.tag!r} holds {model}'s teacher pass ({where}), and the "
                              f"teacher is {teacher}: name the new pass with another --tag")
        if self.L.spec.tag == DEFAULT_TAG and teacher != DEFAULT_TEACHER:
            raise Refused(f"the default tag {DEFAULT_TAG!r} names {DEFAULT_TEACHER}'s pass, and the teacher is "
                          f"{teacher}: name the new pass with --tag")
        self.teacher = teacher

    def dry_run(self, pieces: list[str]) -> None:
        L = self.L
        self.check_teacher()
        h = lambda rel: ("have " if self.have(rel) else "need ") + rel     # noqa: E731
        say(f"=== TEACH {L.graph.name} {L.spec.tag} dry run: teacher {self.teacher} over {self.bench} "
            f"({self.shards} shards), runs {self.name}-*")
        for piece in pieces:
            if piece == "ids":
                say(f"ids:   {h(L.demos_keeps_pass1)}; {h(L.demos_failed_ids)}")
            elif piece == "p2":
                say(f"p2:    {h(self.run_dir(self.name + '-p2') + '/summary.json')}")
            elif piece == "mix":
                say(f"mix:   {h(L.demos_keeps)}; {h(L.demos_detours)}; {h(L.demos_mix)}")
            else:
                g = self.shard(piece)
                say(f"shard {g}: {h(self.run_dir(f'{self.name}-g{g}') + '/summary.json')}")

    def shard(self, piece: str) -> int:
        if not piece.isdigit() or not 0 <= int(piece) < self.shards:
            raise Refused(f"{piece!r} is not a shard (0..{self.shards - 1}), ids, p2 or mix")
        return int(piece)

    def teach(self, run: str, *selection: str) -> None:
        """One pass of the teacher, skipped when its summary exists."""
        if self.have(f"{self.run_dir(run)}/summary.json"):
            say(f"=== SKIP {run} (done)")
            return
        # An OPENAI_BASE_URL would override the routing by model id. It is set
        # empty rather than removed: the client's load_dotenv() fills in only a
        # variable that is absent, so a removed one would come back from .env.
        env = {**self.runner.env, "OPENAI_BASE_URL": ""}
        cmd = [self.runner.python(), "scripts/run_agent_eval.py", "--graph", self.L.graph.name,
               "--model", self.teacher, "--benchmark", self.bench, *selection,
               "--tools", "find,expand,connect", "--max-turns", "20", "--turn-max-tokens", "3000",
               "--api-timeout", "900", "--concurrency", "8", "--no-latency", "--user-note", NOTE, "--name", run]
        log = self.L.root / "logs" / f"{run}.log"
        say(f"=== RUN {run} start {self.runner.now()}\n    {shown(cmd[:-4])} --user-note ... --name {run}")
        rc = self.runner.run(cmd, log, env=env)
        for line in log_lines(log, r"LLM usage|^ALL"):
            say(line[:400])
        say(f"=== RUN {run} end {self.runner.now()}")
        if rc != 0:
            # The leaf's own last words, so a missing key or a refused model
            # is read here and not only in the log.
            for line in log_lines(log, r"Error|error:|Traceback|No API key|Refused", last=3):
                say("    " + line[:300])
            raise Failed(f"{run} failed (exit {rc}; logs/{run}.log)")

    def build(self, cmd: list[str], out: str, last: int) -> None:
        cmd = [self.runner.python(), *cmd]
        say(f"    {shown(cmd)}")
        rc, text = self.runner.capture(cmd)
        for line in text.splitlines()[-last:]:
            say(line)
        if rc != 0 or not self.have(out):
            raise Failed(f"{Path(cmd[1]).name} did not write {out} (exit {rc})")

    def ids(self) -> None:
        """Pass 1's rows, and the ids pass 1 failed, for the second draw."""
        L = self.L
        missing = [g for g in range(self.shards) if not self.have(f"{self.run_dir(f'{self.name}-g{g}')}/summary.json")]
        if missing:
            raise Refused(f"shards {missing} have not run")
        rolls = [a for r in self.shard_runs() for a in ("--rollouts", self.run_dir(r))]
        self.build(["scripts/traj/build_sft_rows.py", *rolls, "--exclude", self.core, "--cap", "1",
                    "--strip-user-note", "--out", L.demos_keeps_pass1], L.demos_keeps_pass1, 12)
        kept = {json.loads(l)["item_id"] for l in open(L.root / L.demos_keeps_pass1) if l.strip()}
        core = {json.loads(l)["id"] for l in open(L.root / self.core) if l.strip()}
        seen: dict[str, str | None] = {}
        for r in self.shard_runs():
            for l in open(L.root / self.run_dir(r) / "results.jsonl"):
                if l.strip():
                    row = json.loads(l)
                    seen[row["id"]] = row.get("tier")
        failed = sorted(i for i in seen if i not in kept and i not in core)
        (L.root / L.demos_failed_ids).write_text("\n".join(failed) + "\n")
        say(f"pass 1: {len(seen)} items rolled, {len(kept)} kept, {len(failed)} failed -> {L.demos_failed_ids} "
            f"{dict(collections.Counter(seen[i] for i in failed))}")

    def p2(self) -> None:
        failed = self.L.root / self.L.demos_failed_ids
        if not failed.exists() or not failed.read_text().strip():
            raise Refused(f"no failed-id list at {self.L.demos_failed_ids}: run 'ids' first")
        self.teach(f"{self.name}-p2", "--ids-file", self.L.demos_failed_ids, "--no-cache")

    def mix(self) -> None:
        L = self.L
        p2 = f"{self.name}-p2"
        if not self.have(f"{self.run_dir(p2)}/summary.json"):
            raise Refused("the second draw is missing: run 'p2' first")
        rolls = [a for r in [*self.shard_runs(), p2] for a in ("--rollouts", self.run_dir(r))]
        self.build(["scripts/traj/build_sft_rows.py", *rolls, "--exclude", self.core, "--cap", "1",
                    "--strip-user-note", "--out", L.demos_keeps], L.demos_keeps, 12)
        self.build(["scripts/traj/build_error_pairs.py", "--rows", L.demos_keeps, "--benchmark", self.bench,
                    "--out", L.demos_detours], L.demos_detours, 5)
        self.build(["scripts/traj/build_substrate_mix.py", "--keeps", L.demos_keeps, "--detours", L.demos_detours,
                    "--out", L.demos_mix], L.demos_mix, 8)

    def execute(self, pieces: list[str]) -> int:
        try:
            self.check_teacher()
            for piece in pieces:
                if piece == "ids":
                    self.ids()
                elif piece == "p2":
                    self.p2()
                elif piece == "mix":
                    self.mix()
                else:
                    g = self.shard(piece)
                    self.teach(f"{self.name}-g{g}", "--shard", f"{g}/{self.shards}")
        except Failed as e:
            say(f"!!! TEACH {e.message}")
            return e.code
        say(f"=== TEACH {self.L.graph.name} {self.L.spec.tag} DONE {self.runner.now()}")
        return 0
