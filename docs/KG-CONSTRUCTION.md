# KG Construction Contract

**Status: PROPOSAL for maintainers' review.**

Scope: rules for *building* a knowledge graph and its evaluation gold that the
harness (`prospector run`, the gate, the metrics) can trust. Read the README's
"Adding a graph" and "Training on your own graph" sections first; this document
assumes them.

Principle: the evaluator is mechanical, so every property it assumes must be
established by a mechanical check *before* evaluation starts. Each rule below
exists because violating it produced a wrong evaluation at least once.

## 1. One renderer, one direction

Every producer of graph text - parsers, extractors, graph builders, gold
writers - emits edges through one shared edge serializer (module path
`graph_reasoner.kg.serialize`, function `render_edge_line`). No second writer
of edge lines anywhere in the tree.

*Why:* gold recorded in traversal order while the KG serializes semantic
direction produces edges that exist in both graphs but never as equal keys;
path recall computed against such gold is bounded below 1.0 by construction and
misreads as a retrieval failure.

*Check:* before any scoring, count gold edges that match only after reversal.
Non-zero means the producer is wrong. For graphs built by your own producers,
reversed matches = 0 is mandatory. A scoring-side reversal tolerance exists
only for legacy or frozen third-party graphs that cannot be re-serialized, and
then only if declared in advance and reported alongside the result. It is never
an escape hatch for a producer bug in a new graph - the fix belongs in the
producer. A silent scoring fix hides the producer bug, and a silent scoring fix
is never allowed.

## 2. Canonical identifiers (joinability)

Node labels are canonical strings in one key space. Every object that *looks*
like a reference to another node must resolve to a subject key, or be declared
a scalar. Sources that name entities relative to their parent (device-tree
child names, short module names, relative paths) are expanded to the full key
space during extraction, not at query time.

*Why:* property objects that name child nodes without the parent path never
join to those nodes' own properties. In one assembled corpus this left zero
joinable pairs, so multi-hop gold generation produced no items at all while
single-hop items looked healthy.

*Check:* report (a) the share of reference-shaped objects that are also
subjects, and (b) the number of candidate 2-hop chains COMPUTED AFTER the
section 4 scaffold filter, over gold-eligible edges only. Chains that exist
before the filter and vanish after it are not candidates; counting them anyway
reports multi-hop capacity the gold will never have. Zero 2-hop candidates
means the corpus cannot satisfy any multi-hop claim - fail the build.

## 3. Hierarchy facts for hierarchical sources

Sources with hierarchy emit parent-to-child containment edges using the same
canonical keys (section 2).

*Why:* containment edges are knowledge in their own right and they are the join
spine that turns isolated subtrees into traversable structure.

## 4. Scaffold versus operational content

Evaluation gold excludes structural scaffolding by an explicit, recorded rule:
symbol tables, root-level cell properties (`#address-cells`, `#size-cells`,
`#clock-cells`), and boilerplate identity fields (`compatible`, `model`,
`phandle`). The KB may retain them; gold may not.

*Why:* scaffolding questions are mechanically valid but trivially answerable
lookups. Inflating a benchmark with them raises pass rates without evidence of
traversal skill, which is the failure mode a retrieval gate exists to detect.

*Ordering:* the filter runs before the section 2 candidate count and before
gold authoring (section 6). The pre-flight receipt (section 7) records the
filter's predicates verbatim - exact property and relation names plus any
regexes - so the gold-eligible edge set can be re-derived from the receipt
alone.

## 5. Provenance: every triple cites its source

Each extracted triple carries file, sha256, line number, and the raw source
line (receipt-derived edges carry a receipt identifier instead), and a checker
can resolve each citation back to the source bytes.

*Check:* citation completeness is 100% before assembly. One uncitable triple
halts the tier for extractor repair. Corpora are never hand-edited to fix a
triple; the extractor is fixed and the tier re-run.

Sidecar row shape for an extraction-derived triple (note the two line numbers:
`line_no` locates the triple in the built KB, `citations[].line` locates the
evidence in the source file):

```json
{"subject": "/thermal-zones/cpu-thermal", "relation": "polling-delay",
 "object": "10000", "provenance_class": "measured", "line_no": 4213,
 "citations": [{"file": "path/to/source.dts",
   "sha256": "<file sha256>", "line": 4213,
   "raw": "polling-delay = <0x2710>;"}]}
```

A receipt-derived edge carries a receipt identifier instead of byte
coordinates; its `citations` may be empty because the provenance is the
receipt itself. Extractors must emit, at minimum, the fields above so a
checker can resolve each citation back to the source bytes without repo
knowledge.

## 6. Gold authoring

- New gold never overwrites frozen gold. Extensions use new ids in a separate
  file so the frozen baseline stays byte-identical and before/after comparisons
  remain valid.
- *Fan-out:* when a question's anchor has many continuations, score the answer
  set the graph reaches - which is what the gate is documented to score against
  - rather than one frozen continuation. Freezing a single continuation among
  dozens mislabels defensible answers as errors.
- *Fan-out diagnostic:* the pre-flight report (section 7) states the
  out-degree distribution - maximum and 95th-percentile target counts per
  (subject, relation) anchor. Any anchor with more than ~10 targets requires
  entity disambiguation or answer-set scoring before gold is authored:
  hub-scale fan-out degrades traversal and citation stability measurably even
  when every edge is correct, and it shows up in gate rates long before the
  per-item metrics explain why.
- *Multi-hop clause:* if the mean gold-path hop count is below 2, either
  re-template the questions or shrink the claim the gate makes. A 1-hop-only
  gold cannot evidence traversal depth.
- Existing drop rules apply (a question that names its own answer; neither end
  of its path present). Added rules: dedupe by question text; deterministic
  selection (sorted iteration, no unseeded randomness); templated items are
  labeled as templated in their metadata.

## 7. Pre-flight gate (mechanical, fail-closed)

Before any evaluation run, all of the following must pass; any failure stops
the run before compute:

1. direction check (reversed matches = 0 for newly produced graphs; a
   pre-declared tolerance only for legacy or frozen third-party graphs),
2. join coverage and 2-hop candidate count computed after the section 4
   filter (section 2), plus the out-degree distribution report with every
   >10-target anchor dispositioned (section 6),
3. citation completeness = 100% (section 5),
4. scaffold filter receipt stating the filter predicates verbatim (section 4),
5. input sha pins recorded for the KB, gold, and evaluator,
6. determinism: re-running extraction and gold generation reproduces the
   outputs byte-for-byte.

## 8. Determinism

Extraction and gold generation are pure functions of pinned inputs: sorted
iteration, fixed templates, stable ordering, and identical bytes on re-run.
Input shas travel alongside outputs so any result can be re-derived from the
exact corpus that produced it.