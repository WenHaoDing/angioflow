# AngioJointField — architecture and data flow

Brief for drawing a schematic figure. Numbers are the trained run
`joint_p1r015_0frame_v1v2_e108f96_gps6_hop2_tok64_wq50_is10_blk3_xf_mh020_pressmax0`
(best.pth, epoch 213). Code: `angioflow/models/joint/model.py`, `.../joint/decoder.py`,
`angioflow/models/interior/encoders.py`.

**One sentence.** A single hierarchical graph-transformer (HierGPS) encodes the vessel
wall geometry *once* into a small set of geometry tokens; two query-point decoders then
read those tokens to predict wall shear stress on the surface and velocity/pressure
inside the lumen, exchanging information with each other at every block.

**Why it is joint.** WSS is the wall trace of the interior velocity gradient
(`τ = μ ∂u/∂n`). The two fields are two views of one flow, so each carries information
the other's supervision never sees directly. The encoder runs once and feeds both heads.

---

## 1. Inputs

| Input | Shape (one case) | Notes |
|---|---|---|
| Wall graph: `pos`, `normals` | `[N_w, 3]` each | N_w ≈ 16 k after cropping |
| Wall `edge_index` | `[2, E]` | triangle mesh edges, symmetrised |
| `geo_dist` | `[N_w, 3]` | geodesic distance to inlet / outlet_0 / outlet_1 |
| `cot_eigvec`, `cot_eigval` | `[N_w, 8]`, `[8]` | cotangent-Laplacian spectral PE |
| Interior query points `pos`, `wall_dist` | `[N_i, 3]`, `[N_i]` | N_i ≈ 45 k sampled in the lumen |
| Time index `t` + cardiac `waveform` | scalar, `[1000, 1]` | the only non-geometric input |

Geometry is static in time, so **the encoder sees no time input at all** — the waveform
enters only at the decoder's query embeddings. Worth showing in the figure: it is why
one encoder pass serves every timestep.

## 2. Stage A — HierGPS geometry encoder (`GeoHierGPSEncoder`)

Node features `[pos, normals, geo_dist]` → linear to 92-d, concatenated with a 16-d
cotangent positional encoding = **108-d** embedding. Edge features are
`[|Δ(pos,normals)|, edge centre]` → 108-d.

```
wall graph  (N_w ≈ 16105 nodes)
   │
   ├─ 2 × GINEConv + LayerNorm                      "pre-convs", full resolution
   │
   ├─ FPS pool, ratio 0.15        →  2416 nodes     ← PANEL 02
   │  rebuild kNN graph, k = 8, made undirected
   │  1 × GINEConv + LayerNorm                      "inter-conv"
   │
   └─ 6 × GPSConv  (GINE local + Performer global attention, 4 heads)
                                  →  2416 geometry tokens, 108-d
```

`n_pool = 1`, so there is exactly **one** coarse level: 16105 → 2416 points (~6.7×
reduction). The kNN graph is symmetrised because a raw kNN graph is directed and ~30 %
of its edges would otherwise have no reverse pair.

## 3. Stage B — latent token resampler (`GeometryLatentResampler`)

2416 variable-length tokens → **64 fixed tokens** (108-d) via learned latent queries
cross-attending to the pooled features, with a positional MLP added to the sources so
*where* geometry occurs is retained. This is the bottleneck the whole figure hinges on:
everything downstream sees the geometry only through these 64 tokens.

## 4. Stage C — two query sets

| | Surface queries | Interior queries |
|---|---|---|
| Module | `WSSQueryEmbed` | `QueryEmbed` |
| Inputs | `pos`, `normals`, `t`, waveform | `pos`, `wall_dist`, `t`, waveform |
| Count | 50 % random subset when training, **all N_w at eval** | all N_i |
| Width | 96-d (`dim_field`) | 96-d |

Fourier features: `n_freqs = 6`; time embedding 8-d; waveform embedding 8-d.

## 5. Stage D — 3 × JointFusionBlock

Each block runs four stages on **both** query sets (VGGT-style alternation):

```
1. GLOBAL       qs ← qs + XAttn(qs, 64 geometry tokens)      4 heads
                qi ← qi + XAttn(qi, 64 geometry tokens)
2. LOCAL        qs ← qs + GATv2 over the 2-hop mesh neighbourhood   (LocalMeshGather)
                qi ← qi + kNN gather of wall pos/normals, k = 16    (LocalSurfaceGather)
2b. FIELD       qs ← qs + GATv2 over its k = 16 nearest INTERIOR states
                qi ← qi + GATv2 over its k = 16 nearest SURFACE states
                edge features: dx, |dx|, dx·n   (n = the wall normal)
3. CROSS-FIELD  qs ← qs + XAttn(qs, all interior states)
                qi ← qi + XAttn(qi, all surface states)
4. FFN          per set, 96 → 192 → 96
```

Stages 2b and 3 are the **joint** part — draw them as a symmetric two-way bridge between
the surface lane and the interior lane. Both directions read the *same snapshot* of the
two sets, so neither sees the other's update within a block (no ordering artefact).
kNN index sets are static geometry, computed once and cached across blocks.

## 6. Stage E — heads and outputs

| Output | Head | Shape | Post-processing |
|---|---|---|---|
| WSS | MLP 96 → 48 → 3 | `[N_w, 3]` | **tangential projection** `τ ← τ − (τ·n) n` |
| Velocity | MLP 96 → 48 → 4, first 3 | `[N_i, 3]` | (no-slip gate available, off in this run) |
| Pressure | same head, last channel | `[N_i, 1]` | zero-max convention |

The tangential projection is a hard physical constraint worth annotating: WSS is a
surface traction, so it cannot have a normal component by construction.

---

## Suggested figure layout

Left → right, three bands:

1. **Input** — `01_surface_mesh.png` (cropped wall mesh, triangulation visible).
2. **Encoder** — `02_pool_L1.png` (FPS points + kNN graph over a ghost of the input)
   → a stack of 6 GPS blocks → a compact row of 64 token squares.
3. **Decoder + outputs** — two parallel lanes (surface / interior) crossed by the
   bidirectional bridge, ×3 blocks, ending in:
   - `03_wss_gt.png` vs `03_wss_pred.png` (shared scale, 0–12 Pa)
   - `04_velocity_gt.png` vs `04_velocity_pred.png` (shared scale)

All panels are transparent PNGs rendered from an identical camera, so they can be
overlaid or aligned. Exact camera and per-panel metadata are in `panels.json`.

Example case shown: `aneug_flow / stable_1`, frame 0 — 16105 wall points, 44544 interior points,
WSS rel-L2 **0.222**, velocity rel-L2 **0.242**.

## Numbers to quote

| Quantity | Value |
|---|---|
| Encoder width `dim_emb` | 108 |
| Decoder width `dim_field` | 96 |
| Cotangent PE modes | 8 (16-d embedding) |
| Pre-convs / inter-convs / GPS layers | 2 / 1 / 6 |
| Pool levels, ratio, kNN k | 1, 0.15, 8 |
| Geometry tokens after resampling | 64 |
| Fusion blocks | 3 |
| Attention heads (global / local) | 4 / 4 |
| Global attention type | Performer |
