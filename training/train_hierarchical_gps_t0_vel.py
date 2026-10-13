"""Train AngioHierarchicalGPS + interior velocity branch (multi-task WSS + velocity).

Design
======

Dataset: AngioFlowWallInteriorDatasetT0 (angioflow/data/datasets_interior.py)
  - surface graph: pos, edge_index, y=WSS, normals, geo_dist, cot PE, ...  (as before)
  - interior pcd:  interior_pos [B,K,3], interior_vel [B,K,3],
                   interior_press [B,K,1], interior_wall_dist [B,K]
    (near-wall-biased pool, K resampled per get() call)

Model: shared trunk + two heads
-------------------------------

  surface graph
      │
  NodeEncoder / EdgeEncoder
      │
  pre-convs (GINEConv, full res)
      │
  SAGPool ×n_pool                          interior_pos + wall_dist
      │                                          │
  GPS layers (coarse tokens) ────────┐      query encoder
      │                              │      (Fourier feats + MLP → [B,K,D])
  unpool + skips                     │           │
      │                              ▼           ▼
      │                         cross-attention decoder
      │                         Q = interior queries
      │                         K,V = coarse GPS tokens (to_dense_batch + mask)
      │                         1-2 blocks (MHA + FFN, pre-LN)
      │                              │
      │                              ├──→ H: interior latents [B,K,D]
      │                              │         │
      │                              │         ▼
      │                              │    field head → vel [B,K,3] (+ press [B,K,1])
      │                              │         │
      ▼                              ▼         ▼
  velocity injection (learned finite difference):

    Physics. WSS = μ·∂u/∂n evaluated at the wall. No-slip gives u(wall) = 0,
    so for an interior point at distance d along the wall normal, first-order
    Taylor:  u(d) ≈ u(0) + d·∂u/∂n = d·∂u/∂n  →  ∂u/∂n ≈ u(d)/d.
    A near-wall velocity sample together with its offset from the wall IS a
    one-sided finite-difference estimate of WSS (up to μ). This is the whole
    reason the velocity branch helps WSS, and why the interior pool is
    near-wall biased.

    Mechanism. Per surface node i (pos x_i, normal n_i): kNN over interior
    points (k≈8). x_j = position of interior neighbour j, Δx = x_j − x_i.
    Edge MLP over, per neighbour:
      vel_pred_j  — the u(d) numerator (branch's predicted velocity at j)
      |Δx|        — the d denominator: distance from j to THIS node i
                    (not j's global wall_dist — the derivative is taken at i)
      Δx·n_i      — offset along i's normal: a point directly above i is a
                    clean ∂u/∂n probe; sideways at the same distance is not
      Δx          — full vector: resolves the tangential direction
                    (WSS is the tangential component of the traction)
      wall_dist_j — j's distance to the NEAREST wall anywhere: flags points
                    whose velocity is boundary-layer-suppressed by some other
                    wall (thin vessel), so slow u there ≠ low shear at i
      H_j         — interior latent: what the 3-channel vel output bottlenecks
                    away (uncertainty, local flow character, pressure context)
    Aggregate over the k neighbours (mean/max) → per-node feature. The MLP can
    realise μ·u(d)/d exactly and learn corrections (curvature, higher-order
    terms, imperfect vel predictions) on top.
      │
      ▼
  concat/fuse with unpooled surface features
      │
  post-convs (full res)
      │
      ▼
  WSS head → [N,3]

Rationale:
  - Branch reads the COARSE GPS tokens (not full-res post-conv features) so the
    velocity gradient flows through the GPS layers — the global module is forced
    to encode flow structure, which is what should help WSS.
  - Cross-attention handles near-wall and far-interior queries with one mechanism;
    tokens are ~pool_ratio**n_pool × N per graph, queries K=4096 → cheap.
  - Velocity → WSS injection is LOCAL (kNN, not attention): WSS = μ·∂u/∂n is a
    local quantity; Δx·n_i gives the finite-difference geometry explicitly, and the
    near-wall-biased pool guarantees close interior neighbours for every wall node.
    Global context already reaches the WSS head via the shared GPS trunk.
  - Gradient reaches the velocity branch twice: from the vel/press loss and from
    the WSS loss through the injection — the WSS head learns how much to trust
    the velocity features even where vel supervision is imperfect.

New modules (angioflow/models/):
  InteriorFieldDecoder
    - fourier_encode(interior_pos) ⊕ wall_dist → MLP → queries [B,K,D]
    - coarse tokens → to_dense_batch → [B,T,D] + key_padding_mask
    - n_xattn × (pre-LN MHA cross-attn + FFN, residual)
    - head: Linear(D → D/2) → ReLU → Linear(D/2 → 4)   # vel(3) + press(1)
    - returns (H, vel_pred, press_pred) — H needed for the injection
  VelocityInjection
    - knn(surface pos → interior pos, k) per graph
    - edge MLP over (H_j, vel_pred_j, Δx, |Δx|, Δx·n_i, wall_dist_j)
      → mean/max aggregate → [N,D]   (see physics rationale above)
    - fuse: x_full = MLP(concat(x_full, agg))
  Combined model wraps the trunk + both modules; trunk forward is split so the
  injection happens between unpool and post-convs.

Losses:
  loss = log_mag_loss(wss) + λ_huber · huber(wss)
       + λ_smooth · smooth_loss(wss, edge_index)
       + λ_vel   · huber(vel_pred,   interior_vel_norm)    # λ_vel ≈ 1.0, co-equal task
       + λ_press · huber(press_pred, interior_press_norm)  # λ_press ≈ 0.1–1.0

Data/transform prerequisites (pending):
  - stats.pt: add vel_mean/vel_std, press_mean/press_std (from train cases)
  - CenterAndRandomRotate: shift interior_pos by the same surface centroid,
    rotate interior_pos AND interior_vel with the same R (vector field!)
  - PosYNormalize: normalize interior_pos with pos stats, vel/press with new stats
  - AneurysmCrop: leaves [1,K,...] tensors alone (first dim ≠ N) — already safe

Eval:
  - main metric unchanged (test WSS logmag/huber/mse_denorm)
  - log vel/press test losses separately; sanity-render predicted vs GT velocity
    magnitude on the interior pcd every ~50 epochs
"""


query (3000) ------------> cross_attn ------> MLP ----> velocity
wall mesh -> self_attn -->

f(x,y,z,manifold) -> vx, vy, vz 

