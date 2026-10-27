"""The policy's latent state at the decision point -> the next move it predicts,
and a direction in it that steers the move (the manuscript's Section 3.6).

Every stage reads stored episodes or the stage before it; nothing here runs a
model except `states` (one teacher-forced pass per stored episode) and `steer`
(whole-episode arms on vLLM with a layer hook). torch and vLLM are imported
inside functions only, so the package imports without them.

    states       stored episodes -> the residual stream after every decoder
                 layer at each turn's read position (index.jsonl, L<n>.npy)
    moves        stored episodes -> the next move of every observation
                 (follow, backtrack, answer, ...) and a knob root's move mix
    readings     reads + moves -> the AUC-by-layer profile of the next-move
                 readings and the Fisher plane of the three moves
    directions   reads + moves at one layer -> the follow and answer axes,
                 random draws and the projected-out axes, with their dose units
    steer        a benchmark through the served policy with a vector added at
                 one layer, one arm per dose, through the ordinary eval
    figure       the figure's data: the plane data, the dose series with the
                 pooled nulls and the per-arm table, for the paper's extractor

The CLIs are `scripts/latent/`; `prospector latent` runs the manuscript's
sequence. Nothing is re-exported: a caller imports each name from the module
that owns it.
"""
