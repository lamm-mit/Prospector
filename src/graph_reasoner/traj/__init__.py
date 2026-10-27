"""Training-row construction: stored rollouts -> the rows the trainers read.

Produces `data/traj/`; the CLIs are `scripts/traj/`. `sft_rows` holds the keep
filter and the supervised row shape, `prefixes` the decision states a decision
round resumes, `error_pairs` the detour rows.

Not to be confused with `bench/`, which goes the other way (graph -> questions)
and shares no code with this package.
"""
from .sft_rows import TRACE_SCHEMA

__all__ = ["TRACE_SCHEMA"]
