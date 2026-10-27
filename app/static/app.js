/* Prospector frontend: replay and live. Renders what a trajectory record
   contains: transcript cards from `messages`, graph growth from per-call
   edge batches, grounding from the server's in_seen flags. No tool semantics
   live here.

   Replay gets a complete record and steps through it. Live gets the same
   record one server-sent event at a time (`start`, a `turn` per tool call
   with that call's new edges, then `final`) and grows the same structures,
   so every control works on a live episode. The server appends the finished
   episode to data/runs, and the URL becomes its replay link.

   The whole-graph map is built once per graph and kept across items and
   modes; an episode only resets the traverse drawn on top. Node positions
   never change. The camera has two framings, the whole map (replay's
   default) and following the traverse (live's default); ⛶ switches. */
"use strict";

const $ = (id) => document.getElementById(id);
const state = {
  mode: null,
  traj: null, step: 0, maxStep: 0, hasFinal: true, playing: null,
  cy: null, bg: null, bgFp: null,
  gctx: null,               // graph context for incremental adds
  fitW: 0,                  // container width when the map was last fitted
  fitWhole: true,           // camera: the whole map, or follow the traverse
  follow: true,             // transcript: keep the newest card in view
  userScrollAt: 0,          // when the reader last scrolled by hand (wheel, touch, drag)
  stopped: false,
  replay: { run: null, item: null },
  live: { es: null, running: false, questions: [], status: null, poll: null, idlePoll: null,
          chain: Promise.resolve(), startedAt: 0, genMs: [], genTok: [], msTotal: 0, tokTotal: 0,
          budgetTouched: false },
};

// Graph colours, on the palette of the paper's figures: teal for a seen edge
// (darker when cited), ochre for what is new this turn and the anchor's
// ring, green for the answer, red for a cited edge never seen, gold for the
// gold path. The map underneath is grey.
const P = {
  mapNode: "#CBD3D2", mapEdge: "#DDE3E2", mapInter: "#E8ECEB",
  edge: "#6FA0A7", fresh: "#B8732A", freshLabel: "#7A4710",
  active: "#6FA3AA", labelText: "#15181A", labelBg: "#FFFFFF",
  backbone: "#FFFFFF", teal: "#0B6875", route: "#7FAAB0", routeArrow: "#6A9AA1",
  anchorBorder: "#9A5A15", anchorFill: "#FFFFFF",
  answer: "#1B6E45", cited: "#0B6875", halluc: "#9C3327",
  gold: "#B08A3E",
};

/* A run that served an adapter chain is labelled Prospector plus the model
   size read off its model string; any other run (an API model, an untrained
   base) by the model string it stored. */
const modelSize = (m) => ((m || "").match(/(\d+(?:\.\d+)?B)(?![\w.])/i) || [])[1] || "";
const modelLabel = (m, adapter) => !adapter ? (m || "model")
  : (modelSize(m) ? `Prospector-${modelSize(m).toUpperCase()}` : "Prospector");
/* The map's size for the meta bar; a large graph is drawn as its
   best-connected nodes, and says so. */
const mapSize = (bg) => bg.n_graph
  ? `map of the ${bg.n.toLocaleString()} best-connected of ${bg.n_graph.toLocaleString()} nodes`
  : `map ${bg.n} nodes / ${bg.m} edges`;

const BLANK = () => ({ calls: [], turn_edges: [], call_focus: [],
                       cited_edges: [], gold_path_edges: [], kb_answer_set: [], anchor: "" });

async function jget(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.json();
}

/* ---------------------------------------------------------------- text */

function esc(s) {
  // Quotes too: the result goes into attributes as well as text.
  return (s || "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

/* The two marks the policy writes, **bold** and `code`, applied after
   escaping so nothing else in its text can become markup. */
function md(s) {
  return esc(s).replace(/\*\*(.+?)\*\*/g, "<b>$1</b>").replace(/`([^`\n]+)`/g, "<code>$1</code>");
}

/* A live record's id is `<item>--<yyyymmdd>-<hhmmss>`: the item and the
   time of day, for display. */
function idParts(id) {
  const m = /^(.*?)--(\d{8})-(\d{2})(\d{2})(\d{2})$/.exec(id || "");
  if (!m) return { base: id || "", time: null };
  return { base: m[1], time: `${m[3]}:${m[4]}` };
}

/* ---------------------------------------------------------------- runs */

async function loadRuns(want) {
  const runs = await jget("/api/runs");
  const sel = $("run-select");
  sel.innerHTML = runs.map(r =>
    `<option value="${encodeURIComponent(r.name)}">${esc(r.name)}</option>`).join("");
  sel.onchange = () => loadItems(sel.value);
  if (!runs.length) return;
  if (want && want.run && runs.some(r => encodeURIComponent(r.name) === want.run)) sel.value = want.run;
  loadItems(sel.value, want && want.item);
}

async function loadItems(run, wantItem) {
  const items = await jget(`/api/runs/${run}/items`);
  const sel = $("item-select");
  sel.innerHTML = items.map(it => {
    // The row's stored verdict (see verdict): ✓, ✗, or "·" if it has none.
    const v = verdict(it);
    const mark = v === null ? "·" : (v ? "✓" : "✗");
    const q = esc((it.question || "").slice(0, 70));
    return `<option value="${encodeURIComponent(it.id)}">${mark} ${esc(it.id)} — ${q}</option>`;
  }).join("");
  sel.onchange = () => loadItem(run, sel.value);
  if (!items.length) return;
  if (wantItem && items.some(it => encodeURIComponent(it.id) === wantItem)) sel.value = wantItem;
  loadItem(run, sel.value);
}

let loadSeq = 0;
function showLoading(msg) { $("loading-text").textContent = msg; $("loading").classList.remove("hidden"); }
function hideLoading() { $("loading").classList.add("hidden"); }

async function loadItem(run, id) {
  stopPlay();
  const seq = ++loadSeq;
  showLoading("loading trajectory…");
  let t;
  try {
    t = await jget(`/api/runs/${run}/items/${id}`);
    if (seq !== loadSeq) return;        // a newer selection superseded this one
    // The map is keyed by the graph fingerprint the trajectory records; an
    // unmatched fingerprint gets no background.
    await prepareGraph(t, t.kg_fingerprint, run);
  } catch (err) {
    if (seq === loadSeq) { hideLoading(); alert(String(err)); }
    return;
  }
  if (seq !== loadSeq) return;
  state.traj = t;
  state.hasFinal = true;
  state.stopped = false;
  state.fitWhole = true;                // a stable frame while scrubbing
  state.maxStep = t.calls.length + 1;   // last step reveals the answer
  state.step = 0;
  state.replay = { run, item: id };
  history.replaceState(null, "", `#replay/${run}/${id}`);   // the item's own link
  const bg = state.bg;
  const meta = `${modelLabel(t.model, t.adapter)} · ${t.n_edges_seen} edges seen · ${t.n_calls} calls · ${t.status}` +
               (bg ? ` · ${mapSize(bg)}` : " · no map (fingerprint unmatched)");
  setMeta(meta);
  showChecks(t.checks);
  buildTranscript(t);
  buildScrubber(t);
  hideLoading();
  setStep(0);
}

function setMeta(text) { $("meta-text").textContent = text; $("meta-text").title = text; }

function showChecks(checks) {
  const badge = $("check-badge");
  if (checks && !checks.attribution_exact) {
    badge.classList.remove("hidden");
    badge.textContent = `attribution inexact (${checks.n_unattributed} unattributed)`;
    badge.title = JSON.stringify(checks);
  } else badge.classList.add("hidden");
}

/* ---------------------------------------------------------- transcript */

function assistantParts(content) {
  const think = [...content.matchAll(/<think>([\s\S]*?)<\/think>/g)].map(m => m[1].trim());
  const calls = [...content.matchAll(/<tool_call>([\s\S]*?)<\/tool_call>/g)].map(m => m[1].trim());
  return { think, calls };
}

function questionCard(t) {
  const { base, time } = idParts(t.id);
  const who = base === "adhoc" ? "your question" : esc(base);
  const tier = t.tier && t.tier !== "adhoc" ? ` · tier ${esc(t.tier)}` : "";
  return `<div class="card q" data-step="0">
    <div class="label">${who}${tier}${time ? ` · ${time}` : ""}</div>
    <div class="question">${md(t.question)}</div></div>`;
}

/* One tool-call card. `total` is the call count in replay and the announced
   budget while live; the label says which. */
function callCard(k, total, meta, assistant, obs, nNew, live) {
  const { think, calls } = assistantParts(assistant || "");
  return `<div class="card" data-step="${k}">
    <div class="label">${live ? `call ${k} · budget ${total}` : `call ${k} of ${total}`}
      <span class="tool-tag tt-${esc(meta.tool || "x")}">${esc(meta.tool || "?")}</span>${meta.ok === false ? ' <span class="err">error</span>' : ""}</div>
    ${think.map(x => `<div class="think">${md(x)}</div>`).join("")}
    ${calls.map(x => `<div class="toolcall">${esc(x)}</div>`).join("")}
    <details class="obs"><summary>observation · +${nNew} edges</summary><pre>${esc(obs || "")}</pre></details>
    <div class="callmeta"><b>+${nNew}</b> new edges · ${meta.cum_edges_seen ?? "?"} seen so far${meta.repeat_of ? " · repeat" : ""}</div>
  </div>`;
}

/* A turn that made no call (malformed or empty), shown at the current step
   with the loop's reminder. */
function noteCard(kind, step, assistant, obs) {
  return `<div class="card note" data-step="${step}">
    <div class="label">${esc(kind)} turn · no call</div>
    ${assistant ? `<div class="toolcall">${esc(assistant)}</div>` : '<div class="think">(empty)</div>'}
    <details class="obs"><summary>reminder</summary><pre>${esc(obs || "")}</pre></details>
  </div>`;
}

/* The stored verdict of a results row: gated_acc, the headline. null means
   ungraded (an ad-hoc live question has no gold); its card is tinted ochre,
   not red. Nothing is recomputed here. */
function verdict(r) {
  r = r || {};
  return r.gated_acc ?? null;
}

function verdictOf(t) {
  return verdict(t.result);
}

function answerCard(t) {
  const cited = t.cited_edges || [];
  const res = t.result || {};
  const v = verdictOf(t);
  const cls = v === null ? "ungraded" : (v ? "" : "wrong");
  const tag = v === null ? "ungraded" : `${v ? "✓" : "✗"} gated_acc`;
  const gate = res.gate_reason !== undefined
    ? `<div class="gate ${res.gate_pass ? "ok" : "bad"}">gate: ${res.gate_pass ? "passed — cited ⊆ seen, answer names a cited node" : esc(res.gate_reason)}</div>` : "";
  return `<div class="card answer ${cls}" data-step="${t.calls.length + 1}">
    <div class="label">final answer · ${esc(t.status)} · ${tag}</div>
    <div class="final">${md(t.answer || "(none)")}</div>
    ${t.gold_answer ? `<div class="gold">gold: ${esc(t.gold_answer)}</div>` : '<div class="gold">no gold — free question, gate only</div>'}
    ${gate}
    <div class="cites">${cited.map(e =>
      `<div class="${e.in_seen ? "ok" : "bad"}">${e.in_seen ? "✓" : "✗ NEVER SEEN"} ${esc(e.source)} —[${esc(e.relation)}]→ ${esc(e.target)}</div>`).join("") || "<span class='gold'>no cited edges</span>"}</div>
  </div>`;
}

function stoppedCard(t) {
  return `<div class="card note" data-step="${t.calls.length}">
    <div class="label">stopped · after ${t.calls.length} calls · nothing saved</div></div>`;
}

/* The header every observation starts with (agent/prompts.py,
   OBSERVATION_HEADER_RE): the call number, the budget and the tool. */
const OBS_HEAD = /OBSERVATION \(call (\d+)\/\d+, \w+\):/;

function buildTranscript(t) {
  const cards = [questionCard(t)];
  // Each assistant turn is paired with the call number in the header of the
  // observation after it, not with its position: a malformed or empty turn
  // gets a reminder and no call record. A turn whose reply has no header
  // made no call and is shown as a note.
  let k = 0;
  const msgs = t.messages || [];
  for (let i = 0; i < msgs.length; i++) {
    const m = msgs[i];
    if (m.role !== "assistant") continue;
    const isLast = !msgs.slice(i + 1).some(x => x.role === "assistant");
    if (isLast) break;                       // final answer gets its own card
    const obs = msgs[i + 1] && msgs[i + 1].role === "user" ? msgs[i + 1].content : "";
    const head = OBS_HEAD.exec(obs);
    if (!head) {
      cards.push(noteCard(m.content ? "malformed" : "empty", k, m.content, obs));
      continue;
    }
    k = +head[1];
    cards.push(callCard(k, t.calls.length, t.calls[k - 1] || {}, m.content, obs,
                        (t.turn_edges[k - 1] || []).length, false));
  }
  cards.push(answerCard(t));
  $("transcript").innerHTML = cards.join("");
}

/* ------------------------------------------------------------ scrubber */

function buildScrubber(t) {
  const seg = $("scrub-seg");
  const v = verdictOf(t);
  const parts = [`<div class="seg seg-q" data-step="0" title="question"></div>`];
  t.calls.forEach((c, i) => {
    const cls = ["find", "expand", "connect"].includes(c.tool) ? c.tool : "find";
    parts.push(`<div class="seg seg-${cls}" data-step="${i + 1}" title="call ${i + 1}: ${esc(c.tool || "?")}"></div>`);
  });
  if (state.hasFinal) {
    const fin = v === null ? "seg-q" : (v ? "seg-final-ok" : "seg-final-bad");
    parts.push(`<div class="seg ${fin}" data-step="${t.calls.length + 1}" title="final answer"></div>`);
  } else if (state.live.running) {
    parts.push(`<div class="seg seg-live" title="generating…"></div>`);
  } else if (state.stopped) {
    parts.push(`<div class="seg seg-stopped" title="stopped"></div>`);
  }
  seg.innerHTML = parts.join("");
  seg.classList.toggle("dense", t.calls.length > 40);   // a long budget is a bar, not buttons
  seg.querySelectorAll(".seg[data-step]").forEach(el =>
    el.onclick = () => { stopPlay(); setStep(+el.dataset.step); });
}

function updateScrubber() {
  document.querySelectorAll("#scrub-seg .seg").forEach(el => {
    const s = +el.dataset.step;
    el.classList.toggle("played", s < state.step);
    el.classList.toggle("current", s === state.step);
  });
}

/* --------------------------------------------------------------- graph */

/* Cytoscape element id, and the key the highlight sets join on. It is the
   raw strings, not the scorer's canonical key (kg.serialize.edge_key, which
   the server uses for in_seen): only highlighting depends on it, never a
   verdict. */
const eid = (e) => `${e.source}␟${e.relation}␟${e.target}`;

/* Elements for one batch of traverse edges at `step`. An endpoint the map
   does not hold (no map loaded, a node off a capped map, or a cited name
   that is not in the graph) is added around the point resetTraverse chose:
   the traverse's centroid, or a fixed point while there is none. */
function traverseEls(edges, step, halluc) {
  const g = state.gctx;
  const els = [];
  const ensureNode = (name) => {
    if (g.placed.has(name)) return;
    g.placed.add(name);
    g.stray += 1;
    els.push({ data: { id: name, label: name, s: 8, step, stray: 1,
                       anchor: name.toLowerCase() === g.anchor ? 1 : 0,
                       answer: g.answers.has(name.toLowerCase()) ? 1 : 0 },
               position: { x: g.cx + 260 * Math.cos(g.stray * 2.4), y: g.cy0 + 260 * Math.sin(g.stray * 2.4) },
               classes: "bg", grabbable: false });
  };
  edges.forEach(e => {
    ensureNode(e.source); ensureNode(e.target);
    if (halluc) {
      els.push({ classes: "hid off",
                 data: { id: "h:" + eid(e), source: e.source, target: e.target,
                         label: e.relation, step, halluc: 1 } });
    } else {
      els.push({ classes: "hid",     // born hidden: no first-paint flash
                 data: { id: eid(e), source: e.source, target: e.target,
                         label: e.relation, step,
                         gold: g.goldKeys.has(eid(e)) ? 1 : 0, cited: 0 } });
    }
  });
  return els;
}

function addTraverse(step, edges) { if (state.cy) state.cy.add(traverseEls(edges, step, false)); }
function addHalluc(edges) { if (state.cy) state.cy.add(traverseEls(edges, state.maxStep, true)); }
function markCited(edges) {
  if (!state.cy) return;
  edges.forEach(e => { const el = state.cy.getElementById(eid(e)); if (el.length) el.data("cited", 1); });
}

/* The map for a graph, or the same one if it is already up: the background
   elements are built once and every episode draws on top. */
async function prepareGraph(t, fp, run) {
  if (!state.cy || state.bgFp !== fp) {
    showLoading("loading the knowledge-graph map…");
    // The run names its own graph file; the server also tries every
    // registered substrate's.
    const url = `/api/graph/${encodeURIComponent(fp)}` + (run ? `?run=${run}` : "");
    const bg = fp ? await jget(url).catch(() => null) : null;
    buildMap(bg, fp);
  }
  resetTraverse(t);
}

function buildMap(bg, fp) {
  if (state.cy) { state.cy.destroy(); state.cy = null; }
  state.bg = bg; state.bgFp = fp;
  const els = [];
  if (bg) {
    for (const n of bg.nodes) {
      els.push({ data: { id: n.id, label: n.id, s: Math.max(12, Math.min(22, n.s)),
                         step: 0, anchor: 0, answer: 0 },
                 position: { x: n.x, y: n.y },
                 classes: "bg", grabbable: false });
    }
    bg.edges.forEach(([a, b], i) => {
      const inter = bg.nodes[a].c !== bg.nodes[b].c;
      els.push({ data: { id: "bg" + i, source: bg.nodes[a].id, target: bg.nodes[b].id },
                 classes: inter ? "bg-e inter" : "bg-e" });
    });
  }
  state.cy = cytoscape({
    container: $("cy"),
    elements: els,
    layout: { name: "preset" },      // positions come from the server; nothing moves
    style: [
      { selector: "node.bg", style: {
          // Cytoscape parses only 3- and 6-digit hex colours; alpha goes in
          // opacity.
          "background-color": P.mapNode, "background-opacity": 0.8,
          "width": "data(s)", "height": "data(s)",
          "z-compound-depth": "bottom",
          "label": "" } },
      { selector: "node.active", style: {
          "background-color": P.active,
          "underlay-color": P.active, "underlay-opacity": 0.14,
          "underlay-padding": 5,
          "label": "", "z-index": 9, "z-compound-depth": "top",
          "transition-property": "background-color",
          "transition-duration": "250ms" } },
      { selector: "node.backbone", style: {
          "shape": "diamond",
          "background-color": P.backbone,
          "border-width": 3, "border-color": P.teal,
          "underlay-color": P.teal, "underlay-opacity": 0.14,
          "underlay-padding": 7, "z-index": 11 } },
      { selector: "edge.route-e", style: {
          "curve-style": "unbundled-bezier",
          "control-point-distances": 130, "control-point-weights": 0.5,
          "line-color": P.route, "width": 2, "opacity": 0.85,
          "target-arrow-shape": "triangle", "target-arrow-color": P.routeArrow,
          "mid-target-arrow-shape": "triangle", "mid-target-arrow-color": P.routeArrow,
          "arrow-scale": 2.6, "label": "",
          "z-index": 12, "z-compound-depth": "top", "events": "no" } },
      { selector: "node.active.lab", style: {
          "label": "data(label)", "color": P.labelText, "font-size": 10.5,
          "text-wrap": "ellipsis", "text-max-width": 140,
          "text-background-color": P.labelBg, "text-background-opacity": 0.88,
          "text-background-padding": 3, "text-background-shape": "roundrectangle",
          "text-valign": "bottom", "text-margin-y": 5, "z-index": 10 } },
      { selector: "node.active[anchor = 1]", style: {
          "border-width": 3, "border-color": P.anchorBorder, "background-color": P.anchorFill } },
      { selector: "node.answer-on", style: {
          "background-color": P.answer, "color": P.answer } },
      { selector: "edge", style: {
          "line-color": P.edge, "width": 1.6, "curve-style": "straight",
          "label": "", "z-index": 8, "opacity": 1,
          "z-compound-depth": "top",
          "transition-property": "line-color, opacity",
          "transition-duration": "250ms" } },
      { selector: "edge.fresh, edge.labelled", style: {
          "label": "data(label)", "font-size": 8.5, "color": P.freshLabel,
          "text-rotation": "autorotate",
          "text-background-color": P.labelBg, "text-background-opacity": 0.88,
          "text-background-padding": 2, "text-background-shape": "roundrectangle" } },
      { selector: "edge.bg-e", style: {
          "curve-style": "haystack", "haystack-radius": 0, "width": 1,
          "line-color": P.mapEdge, "line-opacity": 0.8, "label": "",
          "z-index": 1, "z-compound-depth": "bottom",
          "events": "no", "transition-property": "none" } },
      { selector: "edge.bg-e.inter", style: {
          "width": 0.6, "line-color": P.mapInter, "line-opacity": 0.45 } },
      { selector: "edge.fresh", style: {
          "line-color": P.fresh, "width": 2.2,
          "underlay-color": P.fresh, "underlay-opacity": 0.14, "underlay-padding": 4 } },
      { selector: "edge[gold = 1]", style: { "line-style": "dashed", "line-color": P.gold } },
      { selector: "edge.cited-on", style: {
          "line-color": P.cited, "width": 3,
          "underlay-color": P.cited, "underlay-opacity": 0.14, "underlay-padding": 4 } },
      { selector: "edge.halluc-on", style: { "line-color": P.halluc, "line-style": "dotted" } },
      { selector: "edge.hid", style: { "opacity": 0, "events": "no" } },
      { selector: ".off", style: { "display": "none" } },
    ],
    pixelRatio: 1,
    textureOnViewport: true,
    wheelSensitivity: 0.2,
  });
  // Hover: a node's name, a traverse edge's relation.
  const tip = $("tip");
  state.cy.on("mouseover", "node", (ev) => {
    tip.textContent = ev.target.id();
    tip.classList.remove("hidden");
  });
  state.cy.on("mousemove", "node", (ev) => {
    const r = ev.target.renderedPosition();
    tip.style.left = r.x + "px"; tip.style.top = r.y + "px";
  });
  state.cy.on("mouseout", "node", () => tip.classList.add("hidden"));
  state.cy.on("mouseover", "edge", (ev) => {
    if (ev.target.hasClass("bg-e")) return;
    tip.textContent = ev.target.data("label") || "";
    tip.classList.remove("hidden");
  });
  state.cy.on("mousemove", "edge", (ev) => {
    tip.style.left = ev.renderedPosition.x + "px";
    tip.style.top = ev.renderedPosition.y + "px";
  });
  state.cy.on("mouseout", "edge", () => tip.classList.add("hidden"));
  // Labels are sized in screen pixels. Cytoscape fonts are in model units,
  // so they are recomputed on zoom.
  let rafPending = false;
  state.cy.on("zoom", () => {
    if (rafPending) return;
    rafPending = true;
    requestAnimationFrame(() => { rafPending = false; syncLabels(); });
  });
}

/* Reset everything item-specific on the map: the traverse edges and stray
   nodes go, every map node takes the new record's step, anchor and answer,
   and its batches are drawn. The view is then fitted to the whole map. */
function resetTraverse(t) {
  const cy = state.cy;
  if (!cy) return;
  const bg = state.bg;
  const answers = new Set((t.kb_answer_set || []).map(s => s.toLowerCase()));
  const anchor = (t.anchor || "").toLowerCase();
  const pos = new Map((bg ? bg.nodes : []).map(n => [n.id, n]));
  // step at which each node is first revealed by the traverse
  const nodeStep = new Map();
  t.turn_edges.forEach((batch, k) => batch.forEach(e => {
    for (const n of [e.source, e.target])
      if (!nodeStep.has(n)) nodeStep.set(n, k + 1);
  }));
  const known = [...nodeStep.keys()].filter(n => pos.has(n));
  state.gctx = {
    placed: new Set(pos.keys()), stray: 0, anchor, answers,
    goldKeys: new Set((t.gold_path_edges || []).map(eid)),
    cx: known.length ? known.reduce((s, n) => s + pos.get(n).x, 0) / known.length : 2000,
    cy0: known.length ? known.reduce((s, n) => s + pos.get(n).y, 0) / known.length : 2000,
  };
  cy.stop();
  cy.batch(() => {
    cy.elements().not(".bg-e").not("node.bg").remove();
    cy.nodes("[stray = 1]").remove();
    cy.nodes(".bg").forEach(n => {
      const id = n.id();
      n.data({ step: nodeStep.get(id) ?? 0,
               anchor: id.toLowerCase() === anchor ? 1 : 0,
               answer: answers.has(id.toLowerCase()) ? 1 : 0, label: id });
      n.removeClass("active backbone answer-on lab");
    });
    const els = [];
    t.turn_edges.forEach((batch, k) => els.push(...traverseEls(batch, k + 1, false)));
    els.push(...traverseEls((t.cited_edges || []).filter(e => !e.in_seen), t.calls.length + 1, true));
    cy.add(els);
  });
  markCited((t.cited_edges || []).filter(e => e.in_seen));
  fitMap();
}

function fitMap() {
  if (!state.cy) return;
  state.cy.fit(state.cy.elements(), 30);
  state.fitW = $("cy").clientWidth;
}

/* The camera has two framings and never changes the layout: the whole map,
   or the traverse revealed so far, padded to a minimum box and eased. */
const EASE = { duration: 600, easing: "ease-in-out-cubic" };

function traverseEles() {
  return state.cy.elements().filter(el => el.isNode()
    ? el.hasClass("active")
    : !el.hasClass("bg-e") && !el.hasClass("route-e") && !el.hasClass("hid") && !el.hasClass("off"));
}

function frameTraverse() {
  if (!state.cy) return;
  const tv = traverseEles();
  if (!tv.length) return;
  const bb = tv.boundingBox();
  const MIN_W = 560, MIN_H = 400;           // model units: a few hops' worth of map
  const w = Math.max(bb.w, MIN_W), h = Math.max(bb.h, MIN_H);
  const cx = (bb.x1 + bb.x2) / 2, cy0 = (bb.y1 + bb.y2) / 2;
  const box = { x1: cx - w / 2, y1: cy0 - h / 2, x2: cx + w / 2, y2: cy0 + h / 2, w, h };
  state.cy.stop();
  state.cy.animate({ fit: { eles: box, padding: 70 }, ...EASE });
}

function frameWhole() {
  if (!state.cy) return;
  state.cy.stop();
  state.cy.animate({ fit: { eles: state.cy.elements(), padding: 30 }, ...EASE });
}

// A fit before the pane has a size does not frame the map: refit once it
// has a width. A later resize only tells Cytoscape the new size.
new ResizeObserver(() => {
  if (!state.cy) return;
  state.cy.resize();
  if (state.fitW === 0 && $("cy").clientWidth > 0) { fitMap(); syncLabels(); }
}).observe($("graph-pane"));

function syncLabels() {
  // Labels, backbone diamonds and route arrows keep a constant screen size
  // (Cytoscape styles are in model units). Evidence edge widths are left in
  // model units.
  if (!state.cy) return;
  const z = state.cy.zoom();
  const f = Math.min(140, 13 / z);
  state.cy.batch(() => {
    state.cy.elements(".lab, edge.fresh").forEach(el => {
      el.style({ "font-size": f,
                 "text-background-padding": Math.min(34, 3.2 / z),
                 "text-max-width": Math.min(1500, 170 / z) });
      if (el.isNode()) el.style("text-margin-y", Math.min(46, 5 / z));
    });
    const d = Math.max(14, Math.min(240, 15 / z));
    state.cy.nodes(".backbone").forEach(el =>
      el.style({ "width": d, "height": d,
                 "border-width": Math.max(2, Math.min(30, 2.5 / z)) }));
    state.cy.edges(".route-e").forEach(el =>
      el.style({ "width": Math.max(2, Math.min(26, 2.2 / z)),
                 "control-point-distances": Math.max(60, Math.min(900, 70 / z)) }));
  });
}

function setStep(k) {
  state.step = Math.max(0, Math.min(state.maxStep, k));
  const t = state.traj;
  if (!t || !state.cy) return;
  // "Final" only once the answer exists: while an episode is live, the head
  // step is the latest call, and the answer nodes must not light up early.
  const final = state.hasFinal && state.step === state.maxStep;

  document.querySelectorAll("#transcript .card").forEach(c => {
    const s = +c.dataset.step;
    c.classList.toggle("future", s > state.step);
    c.classList.toggle("current", s === state.step && !c.classList.contains("gen"));
  });
  // Only the current step's observation is open; the rest fold to their
  // edge count.
  document.querySelectorAll("#transcript details.obs").forEach(d => {
    d.open = +d.closest(".card").dataset.step === state.step;
  });
  const cur = document.querySelector("#transcript .card.current");
  if (cur && (!state.live.running || state.follow)) scrollTo(cur);

  const freshNames = new Set();
  if (state.step >= 1 && state.step <= t.calls.length)
    (t.turn_edges[state.step - 1] || []).forEach(e => {
      freshNames.add(e.source); freshNames.add(e.target); });
  // The backbone: nodes the policy operated on, numbered by first call.
  const backbone = new Map();
  (t.call_focus || []).slice(0, Math.min(state.step, t.calls.length)).forEach((fs, k) => {
    (fs || []).forEach(n => { if (!backbone.has(n)) backbone.set(n, k + 1); });
  });
  // The route: arrows thread the backbone nodes in visit order. Only route
  // edges carry arrowheads; an observed edge never does.
  state.cy.remove(".route-e");
  const ordered = [...backbone.entries()].sort((a, b) => a[1] - b[1]).map(x => x[0]);
  for (let i = 0; i + 1 < ordered.length; i++) {
    state.cy.add({ group: "edges", classes: "route-e",
                   data: { id: `route:${i}`, source: ordered[i], target: ordered[i + 1] } });
  }
  state.cy.batch(() => {
    state.cy.elements().forEach(el => {
      if (el.hasClass("bg-e") || el.hasClass("route-e")) return;
      const s = el.data("step");
      if (el.isNode()) {
        const revealed = s > 0 && s <= state.step;
        const bb = backbone.get(el.id());
        el.toggleClass("active", revealed);
        el.toggleClass("backbone", revealed && !!bb);
        el.toggleClass("answer-on", revealed && final && el.data("answer") === 1);
        el.data("label", bb ? `${bb} · ${el.id()}` : el.id());
        // Labelled: backbone nodes (numbered by call), the current step's
        // nodes, the anchor, and the answer at the end.
        el.toggleClass("lab", revealed && (
          !!bb || freshNames.has(el.id()) || el.data("anchor") === 1 ||
          (final && el.data("answer") === 1)));
        return;
      }
      const on = s <= state.step && !(el.data("halluc") && !final);
      el.toggleClass("off", !!el.data("halluc") && !final);
      el.toggleClass("hid", !on);
      el.toggleClass("fresh", on && s === state.step && !el.data("halluc"));
      el.toggleClass("cited-on", final && el.data("cited") === 1);
      el.toggleClass("halluc-on", final && el.data("halluc") === 1);
    });
  });
  updateScrubber();
  syncLabels();
  if (!state.fitWhole) frameTraverse();
  $("step-label").textContent = final
    ? `final · ${t.n_edges_seen} edges · answer`
    : state.step === 0 ? `question`
    : `call ${state.step}/${t.calls.length} (${(t.calls[state.step-1]||{}).tool || ""})`;
}

/* ----------------------------------------------------- transcript scroll */

function scrollTo(el) { el.scrollIntoView({ block: "nearest", behavior: "smooth" }); }
function scrollToEnd() {
  const el = $("transcript");
  el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
}
// While an episode streams, a reader who scrolls up keeps their place and
// the "latest" chip announces new cards. Only a hand scroll changes
// `follow`: the page's own smooth scrolls fire the same scroll events, so
// the decision keys on a recent gesture (wheel, touch or mouse down).
{
  const el = $("transcript");
  const gesture = () => { state.userScrollAt = Date.now(); };
  el.addEventListener("wheel", gesture, { passive: true });
  el.addEventListener("touchmove", gesture, { passive: true });
  el.addEventListener("mousedown", gesture);
  el.addEventListener("scroll", () => {
    if (!state.live.running || Date.now() - state.userScrollAt > 1000) return;
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 48;
    state.follow = atBottom;
    $("latest").classList.toggle("hidden", atBottom);
  });
}
$("latest").onclick = () => { state.follow = true; $("latest").classList.add("hidden"); scrollToEnd(); };

/* ---------------------------------------------------------------- live */

const parseReplayHash = () => {
  const p = location.hash.slice(1).split("/");
  return p[0] === "replay" ? { run: p[1] || null, item: p[2] || null } : null;
};
const modeFromHash = () => (location.hash.startsWith("#replay") ? "replay" : "live");

function setMode(mode) {
  if (mode === state.mode) return;
  state.mode = mode;
  document.querySelectorAll(".mode-btn").forEach(b => b.classList.toggle("on", b.dataset.mode === mode));
  $("replay-controls").classList.toggle("hidden", mode !== "replay");
  $("live-controls").classList.toggle("hidden", mode !== "live");
  $("live-pill").classList.toggle("hidden", mode !== "live");
  // Live is the default, so the hash is set only for replay; an item's own
  // link (#replay/run/item) is kept.
  if (mode === "replay") { if (!location.hash.startsWith("#replay")) history.replaceState(null, "", "#replay"); }
  else history.replaceState(null, "", location.pathname);
  stopPlay();
  clearInterval(state.live.idlePoll);
  clearView(mode === "live" ? "" : "Pick a run and an item to replay.");
  if (mode === "live") liveEnter();
  else { stopLive(); loadRuns(parseReplayHash()).catch(() => {}); }
}

/* Forget the episode on screen. The map stays up; a different graph
   rebuilds it on the next load. */
function clearView(placeholder) {
  if (state.cy) resetTraverse(BLANK());
  state.traj = null; state.step = 0; state.maxStep = 0; state.stopped = false;
  state.follow = true; $("latest").classList.add("hidden");
  $("transcript").innerHTML = placeholder
    ? `<div class="placeholder">${esc(placeholder)}</div>` : "";
  $("scrub-seg").innerHTML = "";
  $("step-label").textContent = "–";
  setMeta(state.bg && state.mode === "live" ? mapMeta() : "");
  $("check-badge").classList.add("hidden");
}

function mapMeta() {
  const st = state.live.status || {};
  const graph = st.graph_label || `${st.backend || ""} knowledge graph`;
  return `${graph} · ${mapSize(state.bg)} · ${modelLabel(st.model, st.adapter)}`;
}

/* The empty state of live mode: four of the benchmark's questions, drawn
   at random, each a one-click ask. */
function renderSamples() {
  const qs = state.live.questions;
  if (state.mode !== "live" || state.traj || !qs.length) return;
  const pick = [...qs].sort(() => Math.random() - 0.5).slice(0, 4);
  $("transcript").innerHTML = `<div class="samples-title">Sample questions</div>` +
    pick.map(q => `<button class="sample" data-id="${esc(q.id)}">${esc(q.question)}<small>${esc(q.id)} · ${esc(q.tier)}</small></button>`).join("");
  $("transcript").querySelectorAll(".sample").forEach(b => b.onclick = () => {
    const q = qs.find(x => x.id === b.dataset.id);
    if (!q) return;
    $("q-input").value = q.question;
    ask();
  });
}

async function liveEnter() {
  try {
    const [st, qs] = await Promise.all([jget("/api/live/status"), jget("/api/live/questions")]);
    state.live.questions = qs;
    $("q-list").innerHTML = qs.map(q =>
      `<option value="${esc(q.question)}">${esc(q.id)} · ${esc(q.tier)}</option>`).join("");
    renderPill(st);
    renderSamples();
    if (st.state === "cold") warmModel();     // start loading now, not on the first ask
    else if (st.state === "loading") pollStatus();
    // Poll while idle, so a second viewer sees that the model is busy.
    state.live.idlePoll = setInterval(async () => {
      if (state.mode !== "live" || state.live.running) return;
      try { renderPill(await jget("/api/live/status")); } catch (_) { /* next tick */ }
    }, 5000);
    await showMap(st);
  } catch (err) {
    renderPill({ state: "error", detail: String(err) });
  }
  $("q-input").focus();
}

/* The map goes up as soon as the server reports the graph's fingerprint,
   before any question and before the model is loaded. */
async function showMap(st) {
  if (state.mode !== "live" || !st.kg_fingerprint || state.traj) return;
  if (state.cy && state.bgFp === st.kg_fingerprint) { if (!$("meta-text").textContent) setMeta(mapMeta()); return; }
  await prepareGraph(BLANK(), st.kg_fingerprint);
  hideLoading();
  if (!state.traj) setMeta(mapMeta());
}

function renderPill(st) {
  state.live.status = st;
  const pill = $("live-pill");
  pill.className = `pill ${st.state}`;
  const chain = (st.adapter || "").split(",").filter(Boolean)
    .map(a => a.split("/").slice(-2).join("/")).join(" → ");
  const size = modelSize(st.model) || "model";
  pill.textContent = {
    cold: `${size} not loaded · click to load`,
    loading: `loading the ${size}… ${st.detail || ""}`,
    ready: `${size} served ${st.driver === "vllm" ? "with vLLM on this machine's GPU" : "with MLX on this Mac"}${st.load_s ? `, ready in ${st.load_s}s` : ""}${st.busy ? " · busy with another viewer's question" : ""}`,
    error: `${st.detail || "error"} · click to retry`,
  }[st.state] || st.state;
  pill.title = `${st.model || ""}\n+ ${chain}\n${st.chat_template || ""} · trained budget ${st.max_turns ?? "?"} calls\nrecords → ${st.run_dir || ""}`;
  const b = $("budget");
  if (!state.live.budgetTouched && st.max_turns) b.value = st.max_turns;
  if (st.max_budget) b.max = st.max_budget;
}

async function warmModel() {
  try { renderPill(await (await fetch("/api/live/load", { method: "POST" })).json()); }
  catch (err) { renderPill({ state: "error", detail: String(err) }); return; }
  pollStatus();
}

function pollStatus() {
  clearInterval(state.live.poll);
  state.live.poll = setInterval(async () => {
    try {
      const st = await jget("/api/live/status");
      renderPill(st);
      showMap(st).catch(() => {});
      if (st.state !== "loading") clearInterval(state.live.poll);
    } catch (_) { clearInterval(state.live.poll); }
  }, 1000);
}

function ask() {
  if (state.live.running) return stopLive("stopped");
  const q = $("q-input").value.trim();
  if (!q) return;
  const match = state.live.questions.find(x => x.question === q);
  const budget = Math.max(1, Math.min(+$("budget").max || 200, parseInt($("budget").value, 10) || 20));
  $("budget").value = budget;
  const url = `/api/live/stream?question=${encodeURIComponent(q)}&max_turns=${budget}` +
              (match ? `&item=${encodeURIComponent(match.id)}` : "");
  startLive(url, q);
}

function startLive(url, question) {
  stopPlay();
  clearView("");
  const t = { id: "…", tier: "?", question, calls: [], turn_edges: [], call_focus: [],
              cited_edges: [], messages: [], gold_path_edges: [],
              kb_answer_set: [], anchor: "", model: "", n_edges_seen: 0, n_calls: 0,
              status: "running", result: null };
  state.traj = t; state.hasFinal = false; state.maxStep = 0; state.step = 0;
  state.fitWhole = false;                 // the camera follows the search
  state.follow = true;
  state.live.running = true; state.live.genMs = []; state.live.genTok = [];
  state.live.msTotal = 0; state.live.tokTotal = 0;
  state.live.chain = Promise.resolve();
  state.live.startedAt = performance.now();
  $("btn-ask").textContent = "Stop"; $("btn-ask").classList.add("stop");
  $("q-input").disabled = true; $("budget").disabled = true;
  $("transcript").innerHTML = questionCard(t);
  showLoading("starting an episode…");
  const es = new EventSource(url);
  state.live.es = es;
  // Events are applied in order, one at a time, even when a handler awaits
  // (the map fetch in `start`): a turn must never race the graph it lands on.
  const on = (name, fn) => es.addEventListener(name, (e) => {
    const data = JSON.parse(e.data);
    state.live.chain = state.live.chain.then(() => fn(data)).catch(err => console.error(name, err));
  });
  on("status", (st) => { renderPill(st); $("loading-text").textContent = st.detail || st.state; });
  on("start", liveStart);
  on("generating", liveGenerating);
  on("delta", liveDelta);
  on("turn", liveTurn);
  on("final", liveFinal);
  on("failure", (d) => { notice(`✗ ${d.message}`); finishLive(); });
  on("end", () => finishLive());
  es.onerror = () => { if (state.live.running) { notice("connection lost"); finishLive(); } };
}

function stopLive(why) {
  if (state.live.es) { state.live.es.close(); state.live.es = null; }
  if (!state.live.running) return;
  const t = state.traj;
  if (why && t) {
    state.stopped = true;
    document.querySelectorAll("#transcript .card.gen").forEach(c => c.remove());
    $("transcript").insertAdjacentHTML("beforeend", stoppedCard(t));
  }
  finishLive();
}

function finishLive() {
  state.live.running = false;
  if (state.live.es) { state.live.es.close(); state.live.es = null; }
  $("btn-ask").textContent = "Ask"; $("btn-ask").classList.remove("stop");
  $("q-input").disabled = false; $("budget").disabled = false;
  $("latest").classList.add("hidden");
  document.querySelectorAll("#transcript .card.gen").forEach(c => c.remove());
  hideLoading();
  if (state.traj && !state.hasFinal) buildScrubber(state.traj);
  jget("/api/live/status").then(renderPill).catch(() => {});
}

function notice(text) {
  $("transcript").insertAdjacentHTML("beforeend", `<div class="notice">${esc(text)}</div>`);
}

function liveMeta(extra) {
  const t = state.traj, st = state.live.status || {};
  const gen = state.live.genMs, tok = state.live.genTok;
  const genS = gen.reduce((a, b) => a + b, 0) / 1000;
  const avg = gen.length ? (genS / gen.length).toFixed(1) : null;     // per tool-call turn
  // The rate covers the whole episode once the final's totals are in;
  // until then, the turns so far.
  const msT = state.live.msTotal || genS * 1000, tokT = state.live.tokTotal || tok.reduce((a, b) => a + b, 0);
  const tps = msT > 0 && tokT ? Math.round(tokT / (msT / 1000)) : null;
  const elapsed = ((performance.now() - state.live.startedAt) / 1000).toFixed(1);
  const graph = st.graph_label || `${st.backend || ""} knowledge graph`;
  setMeta(`${graph} · ${modelLabel(t.model || st.model, t.adapter || st.adapter)} · ${t.chat_template || ""} · budget ${t.max_turns || ""} · ` +
          `${t.n_edges_seen} edges seen · ${t.calls.length} calls` +
          (avg ? ` · ${avg} s/turn` : "") + (tps ? ` · ${tps} tok/s` : "") + ` · ${elapsed} s` +
          (extra ? ` · ${extra}` : ""));
}

async function liveStart(s) {
  const t = state.traj;
  Object.assign(t, { id: s.id, tier: s.tier, question: s.question, anchor: s.anchor || "",
                     gold_answer: s.gold_answer, gold_path_edges: s.gold_path_edges || [],
                     kb_answer_set: s.kb_answer_set || [], kg_fingerprint: s.kg_fingerprint,
                     model: s.model, adapter: s.adapter, chat_template: s.chat_template,
                     max_turns: s.max_turns, messages: s.messages || [], graded: s.graded });
  await prepareGraph(t, s.kg_fingerprint);
  $("transcript").innerHTML = questionCard(t);
  buildScrubber(t);
  liveMeta();
  hideLoading();
  setStep(0);
}

function liveGenerating(g) {
  document.querySelectorAll("#transcript .card.gen").forEach(c => c.remove());
  const label = g.forced ? "final answer · budget spent · generating…" : `call ${g.k} · generating…`;
  $("transcript").insertAdjacentHTML("beforeend",
    `<div class="card gen" data-step="${state.maxStep}"><div class="label">${esc(label)}</div><div class="stream"></div></div>`);
  if (state.follow) scrollTo(document.querySelector("#transcript .card.gen"));
  else $("latest").classList.remove("hidden");
}

function liveDelta(d) {
  const el = document.querySelector("#transcript .card.gen .stream");
  if (el) el.innerHTML = md(d.text);
}

function liveTurn(u) {
  const t = state.traj;
  document.querySelectorAll("#transcript .card.gen").forEach(c => c.remove());
  if (u.gen_ms) state.live.genMs.push(u.gen_ms);
  if (u.n_tokens) state.live.genTok.push(u.n_tokens);
  if (u.kind !== "call") {
    t.messages.push({ role: "assistant", content: u.assistant });
    if (u.observation) t.messages.push({ role: "user", content: u.observation });
    $("transcript").insertAdjacentHTML("beforeend", noteCard(u.kind, state.maxStep, u.assistant, u.observation));
    liveMeta();
    return;
  }
  const atHead = state.step >= state.maxStep;
  const k = u.k;
  t.calls.push(u.call);
  t.turn_edges.push(u.new_edges);
  t.call_focus.push(u.focus || []);
  t.messages.push({ role: "assistant", content: u.assistant });
  if (u.observation) t.messages.push({ role: "user", content: u.observation });
  t.n_calls = k; t.n_edges_seen = u.n_edges_seen;
  // Map nodes this call revealed learn their step now (replay knows it at
  // build; live learns it as the seen set grows).
  if (state.cy) u.new_edges.forEach(e => [e.source, e.target].forEach(n => {
    const el = state.cy.getElementById(n);
    if (el.length && !(el.data("step") > 0)) el.data("step", k);
  }));
  addTraverse(k, u.new_edges);
  $("transcript").insertAdjacentHTML("beforeend",
    callCard(k, t.max_turns, u.call, u.assistant, u.observation, u.new_edges.length, true));
  state.maxStep = k;
  if (u.attribution_agrees === false) {
    const badge = $("check-badge");
    badge.classList.remove("hidden");
    badge.textContent = `call ${k}: parser and seen set disagree`;
  }
  buildScrubber(t);
  liveMeta();
  if (atHead) setStep(k); else updateScrubber();
  if (!state.follow) $("latest").classList.remove("hidden");
}

function liveFinal(f) {
  const t = state.traj;
  document.querySelectorAll("#transcript .card.gen").forEach(c => c.remove());
  Object.assign(t, { status: f.status, answer: f.answer, final_text: f.final_text,
                     cited_edges: f.cited_edges || [], result: f.result,
                     n_edges_seen: f.n_edges_seen, n_calls: f.n_calls, checks: f.checks });
  t.messages.push({ role: "assistant", content: f.final_text });
  state.hasFinal = true;
  state.maxStep = t.calls.length + 1;
  state.live.msTotal = f.gen_ms_total || 0; state.live.tokTotal = f.gen_tokens_total || 0;
  markCited(t.cited_edges.filter(e => e.in_seen));
  addHalluc(t.cited_edges.filter(e => !e.in_seen));
  $("transcript").insertAdjacentHTML("beforeend", answerCard(t));
  showChecks(f.checks);
  buildScrubber(t);
  liveMeta(`${(f.wall_ms / 1000).toFixed(1)} s wall · saved to ${f.saved}`);
  setStep(state.maxStep);
  // The episode is now a stored record; the URL becomes its replay link.
  const run = (f.saved || "").split("/").pop();
  if (run && f.id) history.replaceState(null, "", `#replay/${encodeURIComponent(run)}/${encodeURIComponent(f.id)}`);
}

/* ------------------------------------------------------------ controls */

function stopPlay() {
  if (state.playing) { clearInterval(state.playing); state.playing = null; $("btn-play").textContent = "▶"; }
}
function togglePlay() {
  if (state.playing) return stopPlay();
  if (state.step >= state.maxStep) setStep(0);
  $("btn-play").textContent = "⏸";
  state.playing = setInterval(() => {
    if (state.step >= state.maxStep) return stopPlay();
    setStep(state.step + 1);
  }, 1300);
}

$("btn-reset").onclick = () => { stopPlay(); setStep(0); };
$("btn-prev").onclick = () => { stopPlay(); setStep(state.step - 1); };
$("btn-next").onclick = () => { stopPlay(); setStep(state.step + 1); };
$("btn-play").onclick = togglePlay;
$("btn-fit").onclick = () => {
  state.fitWhole = !state.fitWhole;
  if (state.fitWhole) frameWhole(); else frameTraverse();
};
document.querySelectorAll(".mode-btn").forEach(b => b.onclick = () => setMode(b.dataset.mode));
$("btn-ask").onclick = ask;
$("q-input").addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); ask(); } });
$("budget").addEventListener("input", () => { state.live.budgetTouched = true; });
$("budget").addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); ask(); } });
$("live-pill").onclick = () => {
  const st = state.live.status || {};
  if (st.state === "cold" || st.state === "error") warmModel();
};
// The about panel: a click on ? toggles it, anything else closes it.
$("btn-about").onclick = (e) => {
  e.stopPropagation();
  const open = $("about").classList.toggle("hidden");
  $("btn-about").classList.toggle("on", !open);
};
$("about").addEventListener("click", (e) => e.stopPropagation());
document.addEventListener("click", () => { $("about").classList.add("hidden"); $("btn-about").classList.remove("on"); });
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") { $("about").classList.add("hidden"); $("btn-about").classList.remove("on"); }
  if (["SELECT", "INPUT", "TEXTAREA"].includes(e.target.tagName)) return;
  if (e.key === "ArrowRight") { stopPlay(); setStep(state.step + 1); }
  if (e.key === "ArrowLeft") { stopPlay(); setStep(state.step - 1); }
  if (e.key === "Home") { stopPlay(); setStep(0); }
  if (e.key === "End") { stopPlay(); setStep(state.maxStep); }
  if (e.key === " ") { e.preventDefault(); togglePlay(); }
});

// The hash selects replay and carries the replay item (#replay/run/item).
// A hash set after load is honoured, and so are the browser's back and
// forward.
window.addEventListener("hashchange", () => {
  const mode = modeFromHash();
  if (mode !== state.mode) return setMode(mode);
  if (mode !== "replay") return;
  const w = parseReplayHash();
  if (w && w.run && w.item && (w.run !== state.replay.run || w.item !== state.replay.item))
    loadRuns(w).catch(() => {});
});
// A live link can carry the question and budget (?q=…&budget=…). It fills
// the box and waits for the viewer: a link does not start an episode.
{
  const params = new URLSearchParams(location.search);
  if (params.get("q")) $("q-input").value = params.get("q");
  if (params.get("budget")) { $("budget").value = params.get("budget"); state.live.budgetTouched = true; }
}
setMode(modeFromHash());
