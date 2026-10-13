# Gradient estimation algorithms — comparison

Why this folder exists, separate from `gradient_verification/`: that folder
validated the existing graph-KNN weighted-least-squares stencil
(`models/interior/constraints.build_gradient_stencil`) against Fluent's own
stored gradients, and found it degrades noticeably with fewer neighbors. A
follow-up check (comparing three ways of estimating divergence on the *same*
ground-truth CFD velocity — Fluent's own stored `grad_v` trace, our stencil on
the full un-subsampled pool graph, and our stencil on the sparser graph actually
used during training) found the gap is large even in the best case:

| method | mean\|div\| |
|---|---|
| Fluent's own stored `grad_v`, trace | 1.22 |
| our stencil, full pool graph (degree ~64) | 8.60 |
| our stencil, train-time subgraph (degree ~16) | 21.0 |

That noise lives in the *exported CFD velocity data* (likely cell-to-node
interpolation artifacts from the EnSight export, amplified by numerical
differentiation over short distances) — not something a better algorithm can
undo if it's stuck using that same noisy data as input.

But the continuity loss never needs to differentiate the *ground-truth* field —
only the *model's own predicted* field, which is a continuous, differentiable
function of query position. The model isn't restricted to precomputed graph
neighbors; it can be queried anywhere. So instead of asking a fixed, noisy graph
for neighbors, we can generate our own: place a small number of fresh synthetic
points around a probe (a tiny local element) and query the model there too.

For a locally-linear field, least-squares over >=3 such offsets is exactly the
Green-Gauss / FEM shape-function-derivative gradient — the same family of
method Fluent itself uses internally. A single regular tetrahedron (4 points
around the probe) is the minimal, zero-redundancy case; more well-spread
offsets add back noise-averaging redundancy at the cost of more queries.

## Contents

- `algorithms.py` — the estimators themselves: offset-direction generators
  (regular tetrahedron, Fibonacci-sphere for general K), the shared
  least-squares local-gradient solve, and divergence. Pure tensor math, no
  model dependency.
- `synthetic_benchmark.py` — controlled test against a known analytic
  divergence-free field with injectable noise (calibrated to the ~7-17x noise
  levels measured above), so each method's bias/variance is measurable
  directly against ground truth rather than inferred.
- `model_probe_analysis.py` — the real test: query a trained checkpoint at real
  probe points plus synthetic offsets, sweep the element radius (the classic
  numerical-differentiation step-size tradeoff — too large loses local
  accuracy, too small hits the model's own floating-point noise floor), and
  compare against the existing graph stencil applied to the same predictions.

```bash
conda activate new
export MPLCONFIGDIR=/tmp/angioflow-mpl
python -m angioflow.data.gradient_estimation.synthetic_benchmark
python -m angioflow.data.gradient_estimation.model_probe_analysis \
  --run_name interior_hgps_v1v2_graph_div_r025_d48_b6_div --which best --device cuda:2
```
