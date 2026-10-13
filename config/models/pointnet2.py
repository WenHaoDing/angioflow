"""Model-specific args + builder for AngioPointNet2."""
from __future__ import annotations

from angioflow.models.wss.pointnet2 import AngioPointNet2


def add_args(parser):
    g = parser.add_argument_group("model:pointnet2")
    g.add_argument("--dim_emb",   type=int,   default=64)
    g.add_argument("--base_ch",   type=int,   default=128,
                   help="SA/FP channel width base (c1=base_ch, c2=2x, c3=8x)")
    g.add_argument("--sa_ratio1", type=float, default=0.25, help="FPS ratio for SA1")
    g.add_argument("--sa_ratio2", type=float, default=0.25, help="FPS ratio for SA2")
    g.add_argument("--sa_r1",     type=float, default=0.3,  help="ball radius for SA1 (normalised units)")
    g.add_argument("--sa_r2",     type=float, default=0.5,  help="ball radius for SA2 (normalised units)")
    return parser


def build_model(args, stats, device):
    model = AngioPointNet2(
        n_geo_dist=args.n_geo_dist,
        dim_te=args.dim_te,
        dim_wfe=args.dim_wfe,
        dim_emb=args.dim_emb,
        dim_out=3,
        base_ch=args.base_ch,
        sa_ratios=(args.sa_ratio1, args.sa_ratio2),
        sa_radii=(args.sa_r1, args.sa_r2),
    ).to(device)
    return model
