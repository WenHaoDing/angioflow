# Handoff: predict blood velocity from an aneurysm surface

This is for an agent that already has a 3D lumen surface (for example, from segmented images). It wants the blood velocity inside the vessel around the aneurysm. Use [joint_velocity.py](joint_velocity.py) for this. The older files in this folder (`mesh_to_graph.py`, `run_inference.py` and the others) were written for an older model and data format, and give wrong inputs to this one.

## The model

The joint graph model predicts WSS, velocity and pressure together. You only need the velocity.
- **Training:** about 1,800 cases from three datasets, at frame 0 of a fixed inlet waveform.
- **Status:** still training and not converged. The current bundle is from epoch 40.
- **Accuracy:** on 8 held-out real-patient (AngioFlow) cases, the median relative L2 velocity error is about 0.39 (range 0.21–0.57).
- **Bundle:** one file holding the weights, settings, normalization stats and waveform:
  `/media/yaplab2/wd8tb/wenhao/datasets/AngioFlow-PyG_checkpoints/joint_v3data_0frame_v123_e108f96_gps6_hop2_tok64_wq50_is20_blk3_xf_mh025_pressmax0_fs_warm_b4_lr3e-4_p20_angiotest/deploy_bundle_best.pt`
- **Newer checkpoint:** re-export from the run folder with
  `JointVelocityPredictor.export_bundle(<run_dir>, <out.pt>)`.

**What the velocity means:** it's the field at frame 0 of the training inlet waveform. The outlet flow split is set from outlet diameters with Q0/Q1 = (d0/d1)^2.45. Neither the waveform nor the split is patient-specific.

## Environment

- Use the conda env `new`, with the parent of the `angioflow/` repo on `PYTHONPATH`.
- Use any free GPU except 1 and 2. GPU 1 has faulty memory and GPU 2 overheats. CPU also works, more slowly.
- **Speed:** one case takes about 1 s to prepare, 1–2 s to predict, and 10–15 s more when `predict` samples the lumen points itself.

## What you must provide

1. **An open triangle surface** of the vessel lumen around one aneurysm, in mm (scanner coordinates are fine). It must have exactly one inlet and one or two outlets, each a clean planar cut at a boundary loop. Close every other hole. Training meshes looked like this; stay in this range:

   | | Training (AngioFlow, median, p10–p90) |
   |---|---|
   | Edge length | 0.12 mm (0.11–0.16) |
   | Nodes | ~30k (22k–32k) |
   | Surface area | ~400 mm² (230–630) |
   | Largest extent | ~20 mm (16–24) |

   So clip a local segment of about 15–25 mm around the aneurysm, not a whole vessel tree. Every per-node feature is normalized by the size of the whole surface, so a much larger mesh is out of distribution.

2. **`focused_mask`:** a per-vertex boolean array on your input mesh marking the aneurysm sac and dome. In training this region came from a clipped reconstruction and covered about half the surface. The crop, and so where velocity is predicted, is grown from this region.

3. **`inlet_point`** and **`outlet_points`** (`[p0]` or `[p0, p1]`): a point near each opening, in the same units. Each point picks the opening whose rim centroid is nearest. Your order decides which outlet is outlet_0. That order fixes the geodesic columns and the flow split, so keep it consistent.

## Use

```python
from angioflow.deploy.joint_velocity import JointVelocityPredictor, prepare_surface, save_velocity_sanity

pred = JointVelocityPredictor.from_bundle(BUNDLE, device="cuda:0")
surf = prepare_surface(verts_mm, faces, units="mm", inlet_point=inlet_mm,
                       outlet_points=[out0_mm, out1_mm], focused_mask=sac_mask,
                       remesh=True)   # remesh to 0.12 mm, the training density
print(surf.report)          # nodes, edge_mm, area_mm2, extent_mm, openings: compare with the table
out = pred.predict(surf)    # samples lumen points itself
# out.points [Q,3] mm (your frame), out.velocity [Q,3] m/s
save_velocity_sanity(surf, out, "case_sanity.png")   # look at this before trusting a result
```

- **Your own points:** pass `pred.predict(surf, query_points=pts_mm)`, for example voxel centres of your segmentation. Points outside the crop are dropped; `out.query_index` gives the input row of each result.
- **The sanity PNG:**
  - **Top row:** the labels (inlet, outlets, focused region), the crop, and the mesh numbers next to the training values.
  - **Below:** |v| on three thin slabs through the crop.
  - **Check that** the inlet and outlets sit where you meant them to, and that the focused region covers the sac.
- **Repeat calls:** `predict` is reproducible for a given `seed` (default 0). The pooling adds about 1% noise between seeds, so averaging a few seeds smooths it.

## What `prepare_surface` / `predict` do

This is the same recipe that built the training data, calling the same functions:

| Step | Detail |
|---|---|
| Remesh | pymeshlab isotropic remeshing, 0.12 mm target edge. `focused_mask` is moved to the new vertices by nearest neighbour. Openings are found as boundary loops: the inlet rim is labelled `node_type` 2 and the outlet rims 3 and 4. |
| `pos` | Metres, minus the mesh mean. |
| `normals` | pyvista vertex normals, auto-oriented outward. |
| `geo_dist [N,3]` | Mesh-edge geodesic distance to the inlet, outlet 0 and outlet 1, each scaled to [0, 1]. A missing outlet is filled with 1. |
| `cot_eigvec [N,8]`, `cot_eigval [8]` | The 8 lowest eigenpairs of the cotangent Laplacian, solved as L v = λ M v (`precompute_wall_static.cot_mass_eigs`): unit norm, largest entry positive. Eigenvalues are scaled by A/4π (A = surface area). |
| `flow_split [2]` | (d0/d1)^2.45 from the outlet cap areas, or [1, 0] with one outlet. |
| Crop | Keeps the focused region plus a geodesic ring 0.25 × the farthest geodesic distance deep (training drew 0–0.25). Interior points whose nearest wall was cut away are dropped. |
| Interior points | Inside the capped surface, at least 0.1 mm from the wall, denser near the wall. Density is ~47 points/mm³, the density the model trained at. Each point gets `wall_dist`, the distance to the nearest wall vertex. |
| Normalize | Recentred on the cropped wall. Positions and `wall_dist` are divided by 4.364 mm; the output velocity is multiplied by 0.2832 m/s. |

## Checked so far

[validate_joint_velocity.py](validate_joint_velocity.py) feeds 8 AngioFlow test cases through this path as if they were new patients: the raw CFD wall in mm at a shifted position, plus the focused mask and an inlet point.

| Check | Result |
|---|---|
| Wall features vs training files, given the training's opening points | Geodesics within 6e-8, normals and eigenvectors identical, eigenvalues within 1e-7, flow split identical. The Laplacian is the dataset's own solver (`precompute_wall_static.cot_mass_eigs`), scaled the same way `datasets.py` does. |
| Velocity, deploy vs training pipeline, same points | Differs by 1–4.5% relative. That's within the model's own seed-to-seed noise; the cause is the focused-region geodesic, which correlates 0.999 rather than exactly with the one training seeded from the reconstruction mesh. |
| Query density | Full CFD pool 0.390, a 20% subsample 0.390, sampled lumen points 0.396 (median relative L2). The choice of points barely matters. |
| Remeshed wall | 0.405 median, against 0.390 unremeshed at the same points. The remeshed edge is 0.12 mm. |
| Sanity renders | `deploy/sanity/<case>_cfd_mesh.png` (CFD points, with CFD and error rows) and `<case>_deploy.png` (remeshed wall plus sampled lumen points: the path a new patient takes). |

None of this has been tested on a real segmentation surface: every input was a CFD wall mesh. Check `surf.report` against the table above, and look at `out.velocity`. Peak |v| was about 0.7–0.8 m/s on the test cases.
