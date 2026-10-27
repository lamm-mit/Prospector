"""The prompts of the static arms: the question with a retrieved subgraph
in the prompt (`EVAL_PROMPT`, with its no-think and label variants), or with
nothing (`CLOSED_BOOK_PROMPT`). No tools, no exploration; the citation block
and the ANSWER line are the agentic task's, so one parser and one gate score
both.

`build_eval_prompt` is the one entry. It gives a closed-book item
`CLOSED_BOOK_PROMPT`, so that item is never told to answer only from an empty
edge list. `scripts/arms/run_static_api.py` generates through it and
`scripts/arms/score_static_gated.py` scores the output.
"""
from __future__ import annotations

from ..schema import BenchmarkItem

EVAL_PROMPT = """You are given a knowledge graph and a question. Answer using ONLY \
the information in the graph edges below — do not use outside knowledge.

Graph edges:
{EDGES}

Question: {QUESTION}

Instructions:
1. Reason step by step inside <think>...</think>.
2. Then output a <graph_json>...</graph_json> block containing JSON of the form \
{"nodes": ["..."], "edges": [{"source": "...", "relation": "...", "target": "..."}]} \
listing EXACTLY the edges from the graph above that support your answer, copied \
verbatim (same spelling, same direction).
3. End with a final line: ANSWER: <your concise answer>.

If the graph does not contain enough information, cite what is relevant and say so \
in the answer."""

CLOSED_BOOK_PROMPT = """Answer the question below using your own knowledge.

Question: {QUESTION}

End with a final line: ANSWER: <your concise answer>."""


# The same prompt without the reasoning instruction, for providers that
# restrict asking a model to disclose its reasoning (Anthropic). The citation
# and answer contract is unchanged.
EVAL_PROMPT_NOTHINK = (EVAL_PROMPT
                       .replace("1. Reason step by step inside <think>...</think>.\n", "")
                       .replace("\n2. Then output a <graph_json>", "\n1. Output a <graph_json>")
                       .replace("\n3. End with a final line", "\n2. End with a final line"))
assert "think" not in EVAL_PROMPT_NOTHINK and "1. Output a <graph_json>" in EVAL_PROMPT_NOTHINK


# The label rule: the EM rule is a substring match on the node label, so a
# correct answer in other words is rejected. It is appended as one numbered
# instruction after the ANSWER line's, and the rest of the prompt is unchanged.
# The closed-book prompt cannot take it, since it shows no labels.
LABEL_RULE = ("In ANSWER, name the answer entity exactly as its node label appears in the graph "
              "edges above: copy the spelling, and do not abbreviate, expand, pluralize or rephrase it.")
_ANSWER_LINE = "End with a final line: ANSWER: <your concise answer>."


def _with_label(template: str) -> str:
    head, sep, tail = template.partition(_ANSWER_LINE)
    assert sep, "the ANSWER instruction is the anchor for the label rule"
    n = int(head.rstrip().rsplit("\n", 1)[-1].split(".", 1)[0]) + 1
    return f"{head}{_ANSWER_LINE}\n{n}. {LABEL_RULE}{tail}"


EVAL_PROMPT_LABEL = _with_label(EVAL_PROMPT)
EVAL_PROMPT_NOTHINK_LABEL = _with_label(EVAL_PROMPT_NOTHINK)
assert "4. In ANSWER" in EVAL_PROMPT_LABEL and "3. In ANSWER" in EVAL_PROMPT_NOTHINK_LABEL


def build_eval_prompt(item: BenchmarkItem, *, think: bool = True, label: bool = False) -> str:
    # A closed-book item asks for the graph-free prompt through this meta key
    # (scripts/arms/build_closed_book.py sets it).
    if item.meta.get("perturbation", {}).get("prompt_mode") == "question_only":
        return CLOSED_BOOK_PROMPT.replace("{QUESTION}", item.question)
    template = {(True, False): EVAL_PROMPT, (False, False): EVAL_PROMPT_NOTHINK,
                (True, True): EVAL_PROMPT_LABEL, (False, True): EVAL_PROMPT_NOTHINK_LABEL}[(think, label)]
    return template.replace("{EDGES}", item.subgraph_text).replace(
        "{QUESTION}", item.question
    )
