"""Model-specific args + builder for AngioHierarchicalGPS."""
from __future__ import annotations

from angioflow.models.wss.hierarchical_gps import AngioHierarchicalGPS


def add_args(parser):
    g = parser.add_argument_group("model:hierarchical_gps")
    g.add_argument("--dim_emb",   type=int,   default=108)
    g.add_argument("--n_pre",     type=int,   default=2,    help="GINEConv layers before pooling")
    g.add_argument("--n_gps",     type=int,   default=4,    help="GPS layers on coarsened graph")
    g.add_argument("--n_post",    type=int,   default=4,    help="GINEConv layers after unpool")
    g.add_argument("--n_pool",    type=int,   default=1,    help="FPS pool levels; total compression = pool_ratio**n_pool")
    g.add_argument("--n_inter",   type=int,   default=0,    help="inter-conv layers between pool/unpool stages")
    g.add_argument("--pool_ratio", type=float, default=0.25, help="FPS keep-ratio per level")
    g.add_argument("--pool_knn_k", type=int,   default=6,    help="kNN graph degree rebuilt at each FPS-pooled level")
    g.add_argument("--n_heads",   type=int,   default=4,    help="attention heads in GPSConv")
    g.add_argument("--attn_type", default="performer",      help="GPSConv attention: performer or full")
    g.add_argument("--grad_ckpt", action="store_true",      help="gradient checkpointing on GPS layers")
    g.add_argument("--bidirectional", action="store_true", default=True,
                   help="symmetrize the coarse-level kNN graph rebuilt after each FPS pool "
                        "(a kNN graph is inherently directed); full-resolution data.edge_index "
                        "is symmetrized upstream via the SymmetrizeEdges dataset transform")
    g.add_argument("--no_bidirectional", dest="bidirectional", action="store_false")
    return parser


def build_model(args, stats, device):
    edge_norm_dict = {"mean": stats["edge_mean"].to(device), "std": stats["edge_std"].to(device)}
    model = AngioHierarchicalGPS(
        n_cot=args.lap_k,
        node_dim_in=6,
        n_geo_dist=args.n_geo_dist,
        dim_pe=args.dim_pe,
        dim_te=args.dim_te,
        dim_wfe=args.dim_wfe,
        dim_emb=args.dim_emb,
        n_pre=args.n_pre,
        n_gps=args.n_gps,
        n_post=args.n_post,
        n_pool=args.n_pool,
        n_inter=args.n_inter,
        pool_ratio=args.pool_ratio,
        pool_knn_k=args.pool_knn_k,
        n_heads=args.n_heads,
        dim_out=3,
        pe_n_heads=args.pe_n_heads,
        pe_n_layers=args.pe_n_layers,
        pe_use_mlp=args.pe_use_mlp,
        n_bc=2 if getattr(args, "add_flow_split", False) else 0,   # outlet flow-split BC input
        edge_norm_dict=edge_norm_dict,
        attn_type=args.attn_type,
        grad_ckpt=args.grad_ckpt,
        bidirectional=args.bidirectional,
    ).to(device)
    return model
