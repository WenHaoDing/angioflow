# A geometry → flow foundation model: literature review and design ideas

This note collects what the literature says about pretraining a model that learns the relationship between vessel geometry and blood flow, and turns it into concrete options for AngioFlow.

The model would be used:
- **as a feature extractor**;
- **mainly for one downstream task:** predicting WSS and lumen flow on unseen geometries.

The review was done on 2026-11-10 with the `ml-lit-search` skill (OpenAlex and arXiv; Semantic Scholar was rate-limited). It has two parts:
- the 19 queries from the ablation review (`ABLATIONS.md`), re-screened for foundation-model ideas;
- 17 new queries on foundation models for PDEs and CFD, self-supervised objectives on fields and shapes, multi-modal masking, and inverse design.

All claims about papers below come from their abstracts unless marked otherwise.

## 1. The short answer: which works are most like foundation models?

**From the earlier ablation review:**

| Paper | Why it counts |
|---|---|
| [**MeshMask** (Garnier, Lannelongue, Viquerat, Hachem, 2025)](https://arxiv.org/abs/2501.08738) | **The masked graph network you remembered.** Masks up to 40% of mesh nodes during pretraining, with an asymmetric encoder–decoder (MAE style). State of the art on seven CFD datasets, including a new 3D intracranial aneurysm dataset with >250k nodes per mesh. Pretrains on several datasets at once, which reduces the data needed for new tasks. *Caveat: their downstream task is next-step prediction, not geometry → flow.* |
| [**Inpainting physics** (Weidner, Martin-Ruisanchez, Rückert, Wiestler, 2026)](https://arxiv.org/abs/2605.08832) | **Closest to your idea.** Learns a self-supervised prior over *intracranial aneurysm* velocity fields (masked autoencoder and latent flow matching on spatial tokens). Inference becomes inpainting: fix the known regions (inlet, outlet, unchanged parts of a previous simulation) and fill in the rest. Beats supervised surrogates under boundary-condition and dataset shift, and allows local geometry edits. Steady flow only |
| [AneumoBench (Li et al., 2025)](https://arxiv.org/abs/2505.14717) | 401 aneurysm geometries plus 9,693 locally edited descendants, with 80,752 steady and 9,715 transient CFD cases. Pretraining on the edited descendants, then fine-tuning on the sources, lowers error on held-out geometries. *Shows that cheap geometry variants with CFD are useful pretraining data* |
| STEP (Bai et al., 2026; see `ABLATIONS.md`) | Pretrains on 11,656 *steady* AneuG-Flow cases, then fine-tunes on 449 pulsatile ones: 28% lower error than training from scratch |
| [Garnier et al., ICML 2026](https://arxiv.org/abs/2605.01542) | Their latent representations transfer, when probed, to tasks they weren't trained on (WSS, pressure) |
| Poseidon, UPT / AB-UPT, *Training transformers for mesh-based simulations* | Scaling studies and large pretrained surrogates, but all next-step or single-domain |

**Bottom line.** Hachem's group (MeshMask) and the TUM group (Inpainting physics) are the two aneurysm-specific self-supervised works. Neither does what you describe: **pretraining for geometry → pulsatile WSS + lumen flow, used as a feature extractor for the forward surrogate.** That is an open niche.

## 2. The wider landscape

### 2.1 PDE foundation models (mostly *next-state*, mostly regular grids)

| Paper | Idea | Relevance to us |
|---|---|---|
| [Poseidon (Herde et al., NeurIPS 2024)](https://arxiv.org/abs/2405.19101) | Multiscale operator transformer pretrained on diverse fluid PDEs; 15 downstream tasks; scales with data and model size | Shows pretraining → fine-tuning works for PDEs; time-stepping on grids |
| [Multiple Physics Pretraining (McCabe et al., 2023)](https://arxiv.org/abs/2310.02994) | One transformer predicting many physical systems; shared embedding/normalisation across fields | The idea of normalising heterogeneous fields into one space carries over |
| [DPOT (2024)](https://arxiv.org/abs/2403.03542), [MORPH (2025)](https://arxiv.org/abs/2509.21670), [Unisolver (2024)](https://arxiv.org/abs/2405.17527) | Large multi-PDE pretraining; denoising objective; arbitrary data modalities; PDE components as conditions | Mostly next-state, on grids |
| [CoDA-NO (Rahman et al., NeurIPS 2024)](https://arxiv.org/abs/2403.12553) | Tokenises *channels* (physical variables), allowing self-supervised pretraining across multi-physics systems; +36% in few-shot | **Treating WSS, velocity and pressure as separate token channels** maps directly onto our joint outputs |

**Takeaway:** these show that pretraining helps physics models, but they solve a different problem: predicting the next state on a grid. Little transfers directly beyond the training recipes.

### 2.2 Pretraining for *geometry → field* (closest to your goal)

| Paper | Finding |
|---|---|
| [Cross-Domain Pretraining for Steady-State Neural CFD Surrogates (Zhou, Barati Farimani, Ho, 2026)](https://arxiv.org/abs/2610.10398) | Pretraining across geometries, boundary conditions and fidelities gives **2–3× lower error at the same data size**, or **8× fewer samples for the same error**, than training from scratch. Architecture-agnostic; improves with model size |
| [Toward a Foundation-Model Paradigm for Aerodynamic Prediction (AeroTransformer, 2026)](https://arxiv.org/abs/2604.18062) | Pretrain on ~30k diverse wings, fine-tune with 450 task samples: **84% lower error** than from scratch. The clearest geometry → field pretraining success |
| [Adapting Automotive Aerodynamics Surrogates via Transfer Learning (AB-UPT, 2026)](https://arxiv.org/abs/2605.27968) | **How you fine-tune matters as much as pretraining.** With 20 target samples: full fine-tuning overfits; freezing the encoder fails on unseen shapes; **LoRA adapters** work (R² 0.85) |
| [Geometry Meets Physics: Data-Efficient Pre-Training for Unstructured Neural PDE Solvers (2026)](https://arxiv.org/abs/2610.03363) | For steady problems, pretraining from **intrinsic shape descriptors alone** (no CFD) speeds convergence and improves fine-tuned accuracy |
| [Fine-Scale Feature Encoding in Geometric Pretraining (2026)](https://arxiv.org/abs/2504.20110) | Geometry-only self-supervised representation, decoupled from physics; stresses keeping *fine-scale* detail, which generic shape pretraining tends to lose |
| [Reducing Sensitivity of Neural Physics Simulators to Mesh Topology via Pretraining (2025)](https://arxiv.org/abs/2501.09597) | Autoencoder pretraining makes graph models robust to different meshings of the same object |
| [Geometric Neural Operators (Quackenbush, Atzberger, 2025)](https://arxiv.org/abs/2503.04649) | Pretrained operators that estimate metric and curvature on point clouds, released as reusable geometric feature extractors |
| [How Does Distribution Shift Shape Pretraining Gains? (2026)](https://arxiv.org/abs/2609.20814) | Pretraining is worth 3.25× data at 1,000 target samples, but only ~1.6× at 5,000, and less when the physics differs. **The gain shrinks as target data grows** |
| [Generative Urban Flow Modeling: From Geometry to Airflow with Graph Diffusion (2025)](https://arxiv.org/abs/2512.14725) | Geometry-only conditioned diffusion over unstructured meshes; generalises to unseen geometries; gives uncertainty |

### 2.3 Self-supervised objectives that could be adapted

| Family | Papers | What it offers |
|---|---|---|
| Masked autoencoders on fields | [Masked Autoencoders are PDE Learners (Zhou, Barati Farimani, 2024)](https://arxiv.org/abs/2403.17728); MeshMask; Inpainting physics | Masked reconstruction learns representations that transfer to coefficient regression and to conditioning solvers |
| Masked autoencoders on shapes | [Point-MAE](https://arxiv.org/abs/2203.06604), [Point-M2AE](https://arxiv.org/abs/2205.14401), [MeshMAE](https://arxiv.org/abs/2207.10228), [RI-MAE (rotation-invariant)](https://arxiv.org/abs/2409.00353) | Geometry-only pretraining, needing no CFD |
| **Multi-modal masking** | [**MultiMAE** (Bachmann et al., ECCV 2022)](https://arxiv.org/abs/2204.01678); [MultiMAE for brain MRI: robustness to missing inputs (2026)](https://arxiv.org/abs/2509.11442) | Mask patches *across* several modalities and reconstruct all of them. At test time any subset of modalities can be the input. **This is the template for masking geometry, WSS and velocity together (section 3)** |
| Predicting in latent space (JEPA) | [Point-JEPA](https://arxiv.org/abs/2404.16432); [**AeroJEPA** (2026)](https://arxiv.org/abs/2605.05586); [PI-JEPA (2026)](https://arxiv.org/abs/2604.01349) | AeroJEPA has geometry "context" tokens and predicted *aerodynamic-state* tokens, two latent spaces usable directly for downstream readouts, with an implicit decoder for fields on demand. PI-JEPA pretrains on *unlabelled input fields* with no PDE solves |
| Contrastive / binding | [PICL (2024)](https://arxiv.org/abs/2401.16327); [ImageBind (2023)](https://arxiv.org/abs/2305.05665) | Align geometry and flow embeddings; good for retrieval and probes, weaker for dense prediction |
| Reusable latent spaces | [Self-Supervised Learning of Universal Latent Spaces for Operator Learning (Mishra group, 2026)](https://arxiv.org/abs/2611.05735); [Tokeniser pretraining in physics foundation models (2026)](https://arxiv.org/abs/2603.05598) | Pretrain the *field tokeniser* (an autoencoder) first, then learn the mapping. Gains are largest when pretrained on the same physics |

### 2.4 Cautions from the literature

- [**Specialized Foundation Models Struggle to Beat Supervised Baselines** (2024)](https://arxiv.org/abs/2411.02796): in genomics, satellite imaging and time series, well-tuned supervised models matched or beat the latest foundation models. **Any claim must be against a strong, tuned supervised baseline**, which for us is the current joint model.
- [**Strategies for Pretraining Neural Operators** (2024)](https://arxiv.org/abs/2406.08473): pretraining gains depend heavily on model and data. They are largest with scarce target data, or when the target resembles the pretraining data. Transfer learning and physics-based pretraining worked best.
- The distribution-shift paper above: **gains shrink as labelled data grows.** With ~1,700 pulsatile training cases, our in-distribution gain may be modest. The case for a foundation model is **data efficiency, transfer and interpretable representations**, not a few percent on our own test set.

## 3. Your ideas, analysed

### 3.1 "Mask part of the flow, give the whole geometry, reconstruct the missing flow"

This is sound, and it is **not leakage during pretraining**: completing the field from context is exactly what forces the model to learn how geometry shapes flow. Leakage only matters for the **downstream evaluation**:

- **Hard rule:** the downstream test patients' flow must never appear in pretraining, masked or not.
- **Their geometry:** using it for geometry-only pretraining is transductive, so keep it out too to be safe.

The real risk is a different one: **shortcuts that make reconstruction easy without learning any physics.**

| Shortcut | Why it's a problem | Fix |
|---|---|---|
| **Interpolation** | With small random masks, the missing values are just smooth interpolations of their neighbours | Mask **large, structured regions**: the whole sac, a whole branch, the downstream half, everything beyond a geodesic distance. Use a **high mask ratio** (MAE uses 75%; MeshMask up to 40%) |
| **Cross-field copying** | WSS is the near-wall velocity gradient (μ ∂u/∂n). If the sac's velocity is visible, its WSS follows almost directly, and vice versa | Mask the **same spatial region in every flow field**: WSS on those wall nodes *and* velocity/pressure at nearby lumen points |
| **Time copying** | Across the pulsatile frames, a missing frame is close to the waveform-scaled neighbouring frames | Mask a region across **all** frames, or mask whole frames *and* a region together |
| **Masking with no context at all** | — | Sample the mask ratio up to **100%**. At 100% the task *is* the downstream task (geometry → flow). Pretraining then becomes a continuum that contains the target task, the MultiMAE recipe, and the model works zero-shot as a forward surrogate |

### 3.2 "Give the downstream flow and reconstruct the upstream geometry"

It's a good auxiliary task: it forces flow features to encode *why* the flow looks the way it does. For example, a jet into the sac implies a particular inflow angle and neck. Two catches:

- **The positions give geometry away.** Our flow lives on lumen points and wall nodes, so the *positions* of the visible downstream points reveal the downstream geometry. Fine, since you only ask for the upstream part. But the visible region must not overlap the region being reconstructed, or the point positions hand over the answer.
- **The problem is ill-posed.** Many upstream shapes give similar downstream flow. A regression decoder will predict a blurry average shape; a generative decoder (diffusion or flow matching, as in Inpainting physics) handles this properly. Treat it as a **phase-2 task**: valuable, but harder to build and to evaluate.

A cheaper version of the same idea: from the flow, predict **scalar geometry descriptors** of the masked region (neck width, inflow angle, aspect ratio, stenosis degree). That keeps the "flow → geometry" signal without needing shape generation.

### 3.3 "Diverse tasks so the model has to extract useful features"

Candidate pretraining tasks, roughly from most to least useful for the forward surrogate:

| # | Task | Input → target | Data needed |
|---|---|---|---|
| T1 | **Masked flow completion** | full geometry + partial flow (structured masks; ratio up to 100%) → missing WSS / velocity / pressure | existing pulsatile CFD |
| T2 | **Steady geometry → flow at many inflows** | geometry + inflow rate → steady fields. Learns how the flow pattern changes with flow rate (jets form at higher inflow) | cheap steady CFD (colleague's solver), many geometries |
| T3 | **Masked geometry** | geometry with regions removed → completed geometry (Point-MAE / MeshMAE style) | geometry only: AneuG can generate an **unlimited** supply, no CFD needed |
| T4 | **Cross-field prediction** | e.g. velocity in a region → WSS there, and back. Learns the wall–lumen coupling | existing CFD |
| T5 | **Flow → geometry** (3.2) | downstream flow → upstream geometry or its descriptors | existing CFD |
| T6 | **Latent prediction (JEPA)** | geometry context → latent of masked flow tokens, instead of raw values | any of the above. An option if raw reconstruction wastes capacity on fine detail |

**Data tiers.** This is the single most useful structural idea, and it is what STEP, AneumoBench and AeroTransformer exploit:

| Tier | Data | Scale | Used for |
|---|---|---|---|
| Geometry only | AneuG / AneuGv2 generated shapes | unlimited | T3 |
| Steady CFD | colleague's solver on generated shapes, at several inflow rates | 10⁴ cases | T1, T2, T4 |
| Pulsatile CFD | AneuG-Flow, AneuGv2, AngioFlow | 10³ cases | T1, T4, T5, then fine-tuning |

## 4. How it maps onto our model

Our joint model is already close to a MultiMAE-style backbone:
- a geometry encoder (HierGPS → latent tokens);
- query decoders that output WSS on the wall and velocity/pressure in the lumen.

The changes needed:

1. **A flow input channel with a mask token.** Each wall node and lumen point gets an extra input: the observed field value, or a learned `[MASK]` embedding, plus a field-type flag. This is CoDA-NO's per-channel token idea.
2. **A mask sampler** (structured regions, a ratio drawn up to 100%, consistent across fields and frames, as in 3.1).
3. **Optional heads:** geometry reconstruction (T3), and geometry descriptors from flow (T5).

At fine-tuning, or zero-shot, all flow is masked and the model *is* the forward surrogate, with no architecture change. For the feature-extractor use, the latent tokens are the representation.

Fine-tuning: compare full fine-tuning, a frozen encoder, and **LoRA**. The automotive study found LoRA the only one that worked with very little target data.

## 5. How to show it works (and avoid the "foundation models don't beat supervised" trap)

1. **Data-efficiency curves.** Train the downstream forward model with 5%, 10%, 25%, 50% and 100% of the AngioFlow training patients, both from scratch and from the pretrained model. This is the main figure in every transfer paper. Expect the gap to be largest at small fractions.
2. **Transfer to data we didn't pretrain on.** AnXplore (public, Hachem's group) or AneumoBench, both intracranial aneurysms. Later, possibly other vessels (carotid, aorta).
3. **Probes of "understanding".** Freeze the encoder and fit small linear or MLP readouts for clinically meaningful quantities:
   - jet present or not;
   - impingement zone location;
   - low-WSS area fraction;
   - OSI hotspots;
   - flow acceleration through a stenosis.

   If simple readouts work, that's evidence for the kind of geometry–flow understanding you describe.
4. **Counterfactual edits.** Use the AneuG shape model to change one thing (inflow angle, neck width, sac size, or add a stenosis upstream). Check that the predicted flow changes as physics says it should: does a narrowing upstream raise velocity and WSS downstream? Inpainting physics and AneumoBench both use local geometry edits, so this is an accepted evaluation.
5. **Always against the tuned supervised joint model**, under the same protocol as `ABLATIONS.md`.

## 6. A staged plan

| Phase | What | New data | Decision point |
|---|---|---|---|
| 0 | Current supervised joint model = baseline | none | — |
| 1 | **Masked flow pretraining (T1 + T4) on the existing 1,828 pulsatile cases**: flow input channel, structured masks, ratio up to 100%. Fine-tune and plot data-efficiency curves | none | Pretrain → fine-tune beats from-scratch at small data fractions? If not, revisit the masks before scaling |
| 2 | Add a steady corpus (T2): colleague's solver on many AneuG-generated geometries at several inflow rates | steady CFD, 10⁴ | Gain on curves and transfer beyond phase 1? |
| 3 | Geometry-only pretraining on unlimited AneuG shapes (T3), and flow → geometry (T5) | geometry only | Probes and counterfactual tests improve? |
| 4 | External transfer (AnXplore / AneumoBench), LoRA fine-tuning, release the encoder | public datasets | Paper |

Phase 1 needs no new data and only modest code changes, so it can show early whether the idea pays off.

## 7. Open questions to discuss

1. **Scope:** aneurysm-specific, or all vessels (carotid, coronary, aorta)? "Foundation model" suggests broader, but the data is aneurysm-heavy.
2. **Steady vs pulsatile in pretraining:** are steady solutions plentiful enough (colleague's solver throughput per case) to be the main pretraining corpus?
3. **Which downstream claims matter most:** data efficiency on new hospitals' data, transfer to new vessel types, or interpretable representations? This decides which evaluations to prioritise.
4. **Inflow variation:** all our cases share one waveform, so a pretraining corpus with *varied* inflow (rates and waveforms, which is cheap with steady CFD) is the only way the model can learn inflow dependence.
5. **Positioning:** the paper must state clearly how the work differs from MeshMask (masked pretraining for next-step prediction), Inpainting physics (a steady velocity prior used for inpainting), STEP (steady pretraining, WSS only) and AeroJEPA (aerodynamics).

## Coverage of this review

- **Queries:**
  - 19 from the ablation review, re-screened;
  - 14 on foundation models, pretraining and self-supervised objectives (PDE and CFD foundation models; masked autoencoders on fields, point clouds and meshes; JEPA; contrastive; scaling; inverse design; pretrained geometry encoders);
  - 3 on multi-modal masking and vascular shape pretraining.
- **Search scope:** years 2020–2026; ~450 unique papers screened by title and abstract.
- **Missing source:** Semantic Scholar was rate-limited throughout.
- **Thin area:** vascular-specific self-supervised shape pretraining turned up almost nothing.
- **Not read in full:** any of these papers; the claims come from abstracts. Read MeshMask, Inpainting physics, Cross-Domain Pretraining, AeroJEPA and MultiMAE in full before committing to a design.
