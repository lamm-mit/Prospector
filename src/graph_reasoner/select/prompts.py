"""The verifier prompt, rendered in one place.

Every prompt a verifier is trained on and every prompt it is scored with is
rendered here: if the two differ by a byte the verifier is scored
off-distribution, and the only symptom is a weak AUC. `ANSWER_CAP` and
`EDGE_CAP` are part of the prompt. They bound what the model sees, so changing
them changes the distribution too.
"""
from __future__ import annotations

from ..kg.serialize import render_edge_line

VERIFIER_SCHEMA = "verifier_sft_v1"

YES = "yes"

NO = "no"

ANSWER_CAP = 300   # characters of the answer shown

EDGE_CAP = 12

VERIFIER_PROMPT = """A question was answered by searching a knowledge graph. \
Below is ONE attempt: the answer it gave and the graph edges it cited as \
support.

Question: {question}

Answer given: {answer}
Edges cited:
{edges}

Is this answer correct and fully supported by the cited edges as a connected \
chain from the question's subject to the answer? Reply with only yes or no."""

def edges_of(cited: list[dict]) -> str:
    """The cited chain, one indented edge line each, capped at `EDGE_CAP`; a
    placeholder line when no edge was cited."""
    return "\n".join("    " + render_edge_line(e["source"], e["relation"], e["target"])
                     for e in cited[:EDGE_CAP]) or "    (no edges cited)"

def render_prompt(question: str, answer: str, cited: list[dict]) -> str:
    return VERIFIER_PROMPT.format(question=question, answer=answer[:ANSWER_CAP],
                                  edges=edges_of(cited))
