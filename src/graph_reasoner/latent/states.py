"""Stored episodes -> the residual stream after every decoder layer at each turn's read position.

An episode's transcript is rendered under its concatenative chat template and
run once through the model, teacher-forced, and the residual stream is read
at one position per assistant turn t >= 1: the last token of the generation
header `<|im_start|>assistant\\n`, the state the policy decides its next move
from. Read point 0 is the embedding output and n the output of decoder layer
n. Each row of the reads is labelled from the stored record with the message
it follows (an observation, or a nudge), the call it answers, the tool, and
whether that call brought a gold-path edge (`G`) and which hop (`hop`), the
two flags the labels file carries beside the moves.

Output per run, under `<out>/<run>/`: `shards/<item>.npz` (one per episode),
`index.jsonl` (one row per read, `row` its index in the arrays), `L<n>.npy`
(rows x hidden size, float32, one file per read point), `drops.jsonl` (the
transcripts longer than the token cap) and `meta.json`. The reader is
transformers on torch (`HFReader`); `compare` reports the agreement of two
extractions of one run. `scripts/latent/extract_states.py` is the CLI; torch
and transformers are imported inside functions. Tests: tests/test_latent.py.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import numpy as np

from ..agent.prompts import FORCED_ANSWER, parse_observation_header
from ..kg.serialize import edge_key, normalize
from ..train.tokenize import PLAIN_CHATML, override_chat_template, template_name, token_spans
from .moves import read_records

__all__ = ["HEADER", "HEADER_TOKENS", "gold_hops", "call_arrivals", "decision_rows", "rendering_tokenizer",
           "read_positions", "HFReader", "extract_run", "consolidate", "load_extraction", "compare"]

#: The generation header every assistant turn opens with under plain ChatML
#: (the template's generation prompt, train/tokenize.py), and its token count
#: on the Qwen3.5 vocabulary. The read position is its last token.
HEADER = "<|im_start|>assistant\n"
HEADER_TOKENS = 3
assert PLAIN_CHATML.endswith(HEADER + "{% endif %}"), "HEADER is not plain ChatML's generation prompt"


# ---------------------------------------------------------------- labels

def gold_hops(record: dict) -> list[int | None]:
    """Hop index of each `gold_path_edges` entry, walking the chain from the
    anchor: the edge touching the anchor is hop 1, the edge touching that
    edge's far end is hop 2, and so on. None for an edge the walk never
    reaches (a record whose anchor is off its path)."""
    edges = record.get("gold_path_edges") or []
    ends = [(normalize(e["source"]), normalize(e["target"])) for e in edges]
    hops: list[int | None] = [None] * len(edges)
    cur = normalize(record.get("anchor") or "")
    left = list(range(len(edges)))
    for h in range(1, len(edges) + 1):
        nxt = [j for j in left if cur in ends[j]]
        if not nxt:
            break
        j = nxt[0]
        hops[j] = h
        left.remove(j)
        cur = ends[j][0] if ends[j][1] == cur else ends[j][1]
    return hops


def _gold_key_index(record: dict) -> dict[tuple[str, str, str], int]:
    return {edge_key(e["source"], e["relation"], e["target"]): j
            for j, e in enumerate(record.get("gold_path_edges") or [])}


def call_arrivals(record: dict) -> list[list[int]]:
    """Per call, the `gold_path_edges` indices of the gold edges that call
    added to the seen set: the gold keys among
    `seen_edges[cum_edges_seen(k-1):cum_edges_seen(k)]`. Raises when the
    count disagrees with the stored `n_new_gold`."""
    keys = _gold_key_index(record)
    seen = record["seen_edges"]
    out: list[list[int]] = []
    prev = 0
    for k, c in enumerate(record["calls"], start=1):
        cum = int(c["cum_edges_seen"])
        hits = [keys[edge_key(e["source"], e["relation"], e["target"])]
                for e in seen[prev:cum]
                if edge_key(e["source"], e["relation"], e["target"]) in keys]
        if len(hits) != int(c.get("n_new_gold", 0)):
            raise ValueError(f"{record['id']} call {k}: {len(hits)} gold edges in the "
                             f"seen slice but n_new_gold={c.get('n_new_gold')}")
        out.append(hits)
        prev = cum
    return out


def decision_rows(record: dict) -> list[dict]:
    """One row per assistant turn after the first, in order: `turn` (the
    0-based index among the record's assistant messages), `msg` (its message
    index), `obs_msg` (the message before it) and `kind`: `observation` with
    the call it answers, its tool and the gold-arrival flags, or `forced` for
    the forced-answer message, or `nudge` for any other user message."""
    msgs = record["messages"]
    a_pos = [i for i, m in enumerate(msgs) if m.get("role") == "assistant"]
    calls = record.get("calls") or []
    by_turn = {int(c["turn"]): (k, c) for k, c in enumerate(calls, start=1)}
    arrivals = call_arrivals(record)
    hops = gold_hops(record)
    rows: list[dict] = []
    for t, i in enumerate(a_pos):
        if t == 0:
            continue                      # follows the question, no observation
        prev = msgs[i - 1]
        row = {"id": record["id"], "tier": record.get("tier"), "turn": t, "msg": i, "obs_msg": i - 1}
        if prev.get("role") != "user":
            raise ValueError(f"{record['id']}: assistant turn {t} does not follow a user turn")
        header = parse_observation_header(prev["content"])
        if header is None:
            row["kind"] = "forced" if prev["content"] == FORCED_ANSWER else "nudge"
            rows.append(row)
            continue
        call_no, _announced, tool = header
        # The observation before assistant turn t renders the call turn t-1
        # made; call records count turns from 1, so that call's record carries turn t.
        k, _call = by_turn.get(t, (None, None))
        if k is None or k != call_no:
            raise ValueError(f"{record['id']}: observation after turn {t} says call "
                             f"{call_no} but the call records say {k}")
        new_gold = arrivals[k - 1]
        arriving = [hops[j] for j in new_gold]
        row.update({"kind": "observation", "call": k, "tool": tool,
                    "G": int(len(new_gold) > 0),
                    "hop": min((h for h in arriving if h is not None), default=0) if new_gold else 0})
        rows.append(row)
    return rows


# ---------------------------------------------------------------- positions

def rendering_tokenizer(source: str, chat_template: str | None):
    """The tokenizer that renders the transcripts: loaded from `source`, the
    named template installed when one is given, and refused unless its
    effective template is one of `train/tokenize.CHAT_TEMPLATES` (the
    concatenative renderings a run can be generated under)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(source)
    override_chat_template(tok, chat_template, model=source)
    name = template_name(getattr(tok, "chat_template", None))
    if name is None:
        raise SystemExit(f"{source} carries no named concatenative chat template; pass "
                         f"--chat-template plain_chatml, or point --tokenizer at a trained "
                         f"checkpoint that ships one")
    n = len(tok(HEADER, add_special_tokens=False)["input_ids"])
    if n != HEADER_TOKENS:
        raise SystemExit(f"{source}: {HEADER!r} is {n} tokens, not {HEADER_TOKENS}")
    return tok


def read_positions(messages: list[dict], tokenizer) -> tuple[list[int], dict[int, int]]:
    """(token ids of the whole transcript, {assistant turn t >= 1: read
    position}). The transcript is split with `token_spans`; assistant
    segment t (t >= 1) opens with its generation header, and the read
    position is the header's last token. Each header is checked by decoding.
    """
    p_ids, c_ids, _mask, turn_idx = token_spans(messages, tokenizer)
    ids = list(p_ids) + list(c_ids)
    starts: dict[int, int] = {}
    for i, t in enumerate(turn_idx):
        if t >= 1 and t not in starts:
            starts[t] = len(p_ids) + i
    positions: dict[int, int] = {}
    for t, s in starts.items():
        head = tokenizer.decode(ids[s: s + HEADER_TOKENS])
        if head != HEADER:
            raise ValueError(f"assistant segment {t} opens with {head!r}, not the header")
        positions[t] = s + HEADER_TOKENS - 1
    return ids, positions


# ---------------------------------------------------------------- the reader

class HFReader:
    """The residual stream at chosen positions through transformers: forward
    hooks on the embedding (read point 0) and on every decoder layer, one
    forward of the whole transcript through the decoder without the language
    model head (whose logits over a long transcript would be the pass's
    largest tensor)."""

    engine = "hf"

    def __init__(self, model: str, device: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM

        self.torch = torch
        if device is None:
            device = ("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available() else "cpu")
        self.device = device
        dtype = torch.float32 if device == "cpu" else torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(model, dtype=dtype).to(device).eval()
        self.layers = [m for m in self.model.modules() if type(m).__name__.endswith("DecoderLayer")]
        self.n_layers = len(self.layers)
        self.embed = self.model.get_input_embeddings()
        self.decoder = getattr(self.model, self.model.base_model_prefix)

    def hidden_size(self) -> int:
        return int(self.embed.weight.shape[1])

    def vectors(self, ids: list[int], positions: list[int], layers) -> dict[int, np.ndarray]:
        torch = self.torch
        want = sorted(set(int(l) for l in layers))
        got: dict[int, np.ndarray] = {}
        idx = torch.tensor(positions, dtype=torch.long, device=self.device)

        def keep(li):
            def hook(_m, _inp, output):
                h = output[0] if isinstance(output, tuple) else output
                got[li] = h[0, idx].detach().float().cpu().numpy()
            return hook

        handles = []
        for li in want:
            module = self.embed if li == 0 else self.layers[li - 1]
            handles.append(module.register_forward_hook(keep(li)))
        try:
            with torch.no_grad():
                self.decoder(input_ids=torch.tensor([ids], device=self.device), use_cache=False)
        finally:
            for hd in handles:
                hd.remove()
        return {li: got[li] for li in want}

    def release(self) -> None:
        if self.device == "mps":
            self.torch.mps.empty_cache()
        elif self.device.startswith("cuda"):
            self.torch.cuda.empty_cache()


# ---------------------------------------------------------------- the run

def shard_path(out: Path, item_id: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.#-]", "_", item_id)
    return out / "shards" / f"{safe}.npz"


def extract_run(run: Path, out: Path, reader, tokenizer, layers, *, max_tokens: int,
                limit: int | None = None, ids: set[str] | None = None, log=print) -> None:
    """Read every episode of `run` into `out`: a shard per episode (an
    existing shard is kept), then the consolidated index and arrays."""
    run, out = Path(run), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "shards").mkdir(exist_ok=True)
    records = read_records(run)
    if ids:
        records = [r for r in records if r["id"] in ids]
    if limit:
        records = records[:limit]
    drops: list[dict] = []
    kept = 0
    t_start = time.time()
    tokens_done = 0
    for n, rec in enumerate(records, start=1):
        sp = shard_path(out, rec["id"])
        if sp.exists():
            kept += 1
            continue
        rows = decision_rows(rec)
        tok_ids, positions = read_positions(rec["messages"], tokenizer)
        if len(tok_ids) > max_tokens:
            drops.append({"id": rec["id"], "tokens": len(tok_ids), "max_tokens": max_tokens})
            continue
        for r in rows:
            r["pos"] = positions[r["turn"]]
            r["tokens"] = len(tok_ids)
        t0 = time.time()
        vecs = reader.vectors(tok_ids, [r["pos"] for r in rows], layers)
        tokens_done += len(tok_ids)
        payload = {f"L{l}": v for l, v in vecs.items()}
        payload["rows"] = np.array(json.dumps(rows))
        tmp = sp.with_suffix(".tmp.npz")
        np.savez(tmp, **payload)
        os.replace(tmp, sp)
        kept += 1
        if n % 10 == 0 or n == len(records):
            el = time.time() - t_start
            log(f"  {run.name}: {n}/{len(records)} episodes, {len(tok_ids)} tok in "
                f"{time.time() - t0:.1f}s, {tokens_done / max(el, 1e-9):.0f} tok/s overall")
        if n % 20 == 0:
            reader.release()
    with (out / "drops.jsonl").open("w", encoding="utf-8") as f:
        for d in drops:
            f.write(json.dumps(d) + "\n")
    consolidate(out, records, layers)
    meta = {"run": str(run), "engine": reader.engine, "layers": sorted(int(l) for l in layers),
            "max_tokens": max_tokens, "episodes": kept, "dropped": len(drops),
            "hidden_size": reader.hidden_size(), "n_layers": reader.n_layers,
            "header": HEADER, "header_tokens": HEADER_TOKENS}
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    log(f"  {run.name}: {kept} episodes kept, {len(drops)} dropped over {max_tokens} tokens")


def consolidate(out: Path, records: list[dict], layers) -> None:
    """Shards -> `index.jsonl` and one `L<layer>.npy` per layer, in record
    order, skipping episodes without a shard (dropped)."""
    out = Path(out)
    rows_all: list[dict] = []
    arrays: dict[int, list[np.ndarray]] = {int(l): [] for l in layers}
    for rec in records:
        sp = shard_path(out, rec["id"])
        if not sp.exists():
            continue
        with np.load(sp) as z:
            rows = json.loads(str(z["rows"]))
            for l in arrays:
                arrays[l].append(z[f"L{l}"])
        rows_all.extend(rows)
    with (out / "index.jsonl").open("w", encoding="utf-8") as f:
        for i, r in enumerate(rows_all):
            r["row"] = i
            f.write(json.dumps(r) + "\n")
    for l, parts in arrays.items():
        arr = np.concatenate(parts, axis=0) if parts else np.zeros((0, 0), dtype=np.float32)
        np.save(out / f"L{l}.npy", arr)


def load_extraction(out: Path) -> tuple[list[dict], dict[int, np.ndarray]]:
    """(index rows, {layer: array}) of a consolidated extraction directory."""
    out = Path(out)
    rows = [json.loads(l) for l in (out / "index.jsonl").open(encoding="utf-8") if l.strip()]
    arrays = {int(p.stem[1:]): np.load(p) for p in sorted(out.glob("L*.npy"))}
    return rows, arrays


def compare(a: Path, b: Path, log=print) -> dict:
    """Per-layer agreement of two extractions of the same run on the rows
    they share (by item and turn): the cosine and the relative L2 error, as
    medians and extremes. The check of one reader against another."""
    rows_a, arr_a = load_extraction(a)
    rows_b, arr_b = load_extraction(b)
    key = lambda r: (r["id"], r["turn"])  # noqa: E731
    ib = {key(r): r["row"] for r in rows_b}
    pairs = [(r["row"], ib[key(r)]) for r in rows_a if key(r) in ib]
    if not pairs:
        raise SystemExit("the two extractions share no rows")
    ia, jb = zip(*pairs)
    report = {"rows": len(pairs), "layers": {}}
    for l in sorted(set(arr_a) & set(arr_b)):
        x, y = arr_a[l][list(ia)], arr_b[l][list(jb)]
        nx_, ny = np.linalg.norm(x, axis=1), np.linalg.norm(y, axis=1)
        cos = np.sum(x * y, axis=1) / np.maximum(nx_ * ny, 1e-12)
        rel = np.linalg.norm(x - y, axis=1) / np.maximum(ny, 1e-12)
        report["layers"][l] = {"cos_min": float(cos.min()), "cos_median": float(np.median(cos)),
                               "rel_max": float(rel.max()), "rel_median": float(np.median(rel))}
        log(f"  L{l:>2}: cosine median {np.median(cos):.5f} min {cos.min():.5f}; "
            f"relative error median {np.median(rel):.4f} max {rel.max():.4f}")
    return report
