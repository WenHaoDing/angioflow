"""CLI arg groups shared by every model in `angioflow.training.train`.

Model-specific architecture args (widths, depths, attention type, ...) live in
`angioflow/config/models/<name>.py` instead — see that package's `add_args`.
"""
from __future__ import annotations


def add_data_args(parser):
    g = parser.add_argument_group("data")
    g.add_argument("--case_root_angioflow", default="/media/yaplab2/wd8tb/wenhao/datasets/angioflowv2_merged")
    g.add_argument("--case_root_aneug_flow", default="/media/yaplab2/wd8tb/wenhao/datasets/angioflowv1/transient_data")
    g.add_argument("--save_dir",     default="/media/yaplab2/wd8tb/wenhao/datasets/angioflow_pyg_v2")
    g.add_argument("--extracted", action="store_true",
                   help="--save_dir is a downloaded AngioFlow-PyG root: find cases from its files "
                        "(all datasets incl. aneugv2), per-dataset layout, never process")
    g.add_argument("--n_frames",     type=int,   default=80)
    g.add_argument("--decimate_ratio", type=float, default=0.5)
    g.add_argument("--decimate_method", choices=["quadric", "fps"], default="quadric")
    g.add_argument("--lap_k",        type=int,   default=8)
    g.add_argument("--waveform_path", default="/media/yaplab2/wd8tb/wenhao/datasets/angioflowv2_merged/_meta/waveform_yiying.txt")
    g.add_argument("--train_frac",   type=float, default=0.8)
    g.add_argument("--exclude_anomalies", action="store_true", default=True)
    g.add_argument("--no_exclude_anomalies", dest="exclude_anomalies", action="store_false")
    g.add_argument("--versions",     default=None,
                   help="comma-separated data versions to include, e.g. 'aneug_flow' or 'aneug_flow,angioflow'")
    g.add_argument("--frames",       default="0",
                   help="comma-separated frame ids (e.g. '0' or '0,20,40,60') or 'all'")
    g.add_argument("--mask_low",     type=float, default=0.0)
    g.add_argument("--mask_high",    type=float, default=0.25)
    g.add_argument("--crop_aneurysm_only", action="store_true",
                   help="deterministic crop to the aneurysm region (global_geo_dist == 0); "
                        "disables the random parent-vessel ring")
    g.add_argument("--stats_n_cases", type=int, default=100)
    g.add_argument("--stats_frames_per_case", type=int, default=8)
    return parser


def add_encoder_args(parser):
    """Shared NodeEncoder conditioning args (time / waveform / positional encoding)."""
    g = parser.add_argument_group("encoder")
    g.add_argument("--n_geo_dist",  type=int, default=3)
    g.add_argument("--dim_pe",      type=int, default=16)
    g.add_argument("--dim_te",      type=int, default=8)
    g.add_argument("--dim_wfe",     type=int, default=8)
    g.add_argument("--pe_n_heads",  type=int, default=2)
    g.add_argument("--pe_n_layers", type=int, default=2)
    g.add_argument("--pe_use_mlp",  action="store_true", default=True,
                   help="MLP (not transformer) in CotPE — lighter on GPU")
    return parser


def add_loss_args(parser):
    g = parser.add_argument_group("loss")
    g.add_argument("--max_wss",       type=float, default=35.0)
    g.add_argument("--log_eps",       type=float, default=0.1)
    g.add_argument("--pointwise_loss", choices=["huber", "mse"], default="huber",
                   help="pointwise regression loss on normalized predictions")
    g.add_argument("--lambda_huber",  type=float, default=5.0,
                   help="weight on the pointwise (huber or mse) loss")
    g.add_argument("--lambda_smooth", type=float, default=0.1,
                   help="weight for TV smoothness regulariser on predictions")
    g.add_argument("--lambda_logmag", type=float, default=0.1,
                   help="weight for log-magnitude loss on predictions")
    g.add_argument("--lambda_dir",    type=float, default=0.1,
                   help="weight for directional loss: |wss_pred . n| in Pa/s "
                        "(WSS must be tangent to the wall)")
    return parser


def add_optim_args(parser):
    g = parser.add_argument_group("optim")
    g.add_argument("--lr",            type=float, default=1e-3)
    g.add_argument("--lr_end_factor", type=float, default=0.01)
    g.add_argument("--total_epochs",  type=int,   default=500)
    g.add_argument("--batch_size",    type=int,   default=16)
    g.add_argument("--num_workers",   type=int,   default=4)
    g.add_argument("--grad_clip_norm", type=float, default=1.0)
    return parser


def add_schedule_args(parser):
    """LR schedule: reduce_on_plateau vs cosine, shared by train.py and train_interior.py
    (each script keeps its own default -- see their respective arg groups for why)."""
    g = parser.add_argument_group("lr_schedule")
    g.add_argument("--lr_schedule", choices=["cosine", "reduce_on_plateau", "linear", "step"],
                   default="reduce_on_plateau",
                   help="'reduce_on_plateau' holds LR constant until test_loss stops improving "
                        "for --lr_plateau_patience epochs, then cuts it by --lr_plateau_factor -- "
                        "lets a productive LR 'dwell' as long as it's actually still helping, "
                        "instead of a fixed schedule guessing epoch boundaries (validated in "
                        "train_interior.py, now the default here too); 'cosine' = plain "
                        "CosineAnnealingLR(T_max=total_epochs); 'linear' = LinearLR, "
                        "lr -> lr*lr_end_factor over total_epochs; 'step' = StepLR, halved "
                        "every 20% of total_epochs")
    g.add_argument("--lr_warmup_epochs", type=int, default=5,
                   help="linear LR warmup 0->lr over this many epochs before either schedule "
                        "kicks in (0 disables)")
    g.add_argument("--lr_plateau_factor", type=float, default=0.5,
                   help="[reduce_on_plateau] LR multiplier applied when test_loss plateaus")
    g.add_argument("--lr_plateau_patience", type=int, default=15,
                   help="[reduce_on_plateau] epochs with no test_loss improvement before "
                        "cutting LR")
    return parser


def add_misc_args(parser):
    g = parser.add_argument_group("misc")
    g.add_argument("--device",        default="cuda:0")
    g.add_argument("--seed",          type=int, default=0)
    g.add_argument("--run_name",      default="run0")
    g.add_argument("--ckpt_root",     default="/media/yaplab2/wd8tb/wenhao/datasets/angioflow_checkpoints",
                   help="root dir for checkpoints; run saves to <ckpt_root>/<run_name>/")
    g.add_argument("--resume",        action="store_true", default=True)
    g.add_argument("--wandb",         action="store_true")
    g.add_argument("--wandb_entity",  default=None)
    g.add_argument("--wandb_project", default="AngioFlow")
    g.add_argument("--wandb_run_id",  default=None)
    return parser
