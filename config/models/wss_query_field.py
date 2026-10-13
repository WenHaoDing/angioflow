"""Model-specific args + builder for AngioWSSQueryField (query-point-style WSS baseline)."""
from __future__ import annotations

from angioflow.models.wss import build_wss_query_field


def add_args(parser):
    g = parser.add_argument_group("model:wss_query_field")
    g.add_argument("--backbone",   choices=["hgps", "pointnet"], default="hgps")
    g.add_argument("--dim_emb",    type=int, default=128, help="geometry encoder / token width")
    g.add_argument("--dim_field",  type=int, default=128, help="cross-attention decoder width")
    g.add_argument("--n_blocks",   type=int, default=2,   help="two-scale fusion blocks (global+local)")
    g.add_argument("--xattn_heads", type=int, default=4,  help="heads in the global cross-attention")
    g.add_argument("--n_freqs",    type=int, default=6,   help="Fourier frequencies for query pos")
    g.add_argument("--geometry_token_pool", choices=["none", "latent"], default="none",
                   help="global tokens: variable downsampled graph (none) or fixed learned latents")
    g.add_argument("--n_geometry_tokens", type=int, default=64,
                   help="fixed tokens per graph when --geometry_token_pool latent")
    g.add_argument("--token_pool_heads", type=int, default=4)
    g.add_argument("--token_pool_layers", type=int, default=1)
    g.add_argument("--query_sample_ratio", type=float, default=1.0,
                   help="fraction of wall nodes to decode/supervise per training step "
                        "(1.0 = every node, matching the U-Net baselines); the geometry "
                        "encoder still always sees the FULL mesh regardless -- this only "
                        "controls the decoder's query count. Full resolution is always "
                        "used at eval/inference time, whatever this is set to")
    g.add_argument("--query_max_points", type=int, default=None,
                   help="cap on queried nodes/graph per step (ceiling on "
                        "query_sample_ratio; bounds memory on the largest meshes)")
    g.add_argument("--local_hops", type=int, default=2,
                   help="LocalMeshGather's receptive-field radius: union of 1..N hop "
                        "REAL mesh neighbours (never a spatial kNN, so it can't cross "
                        "onto a different, spatially-close-but-disconnected branch, e.g. "
                        "across a thin bifurcation ridge). Edge count grows ~quadratically "
                        "with this, so it's the main local-branch GPU-cost knob")
    g.add_argument("--local_heads", type=int, default=4,
                   help="attention heads in LocalMeshGather's GATv2 layer (only used "
                        "when --local_module gat)")
    g.add_argument("--local_module", choices=["gat", "edge_mlp"], default="gat",
                   help="local-branch aggregator: 'gat' = GATv2 attention (default, "
                        "query-dependent neighbour weighting); 'edge_mlp' = cheaper "
                        "per-edge MLP + mean-pool, no attention (the original design, "
                        "kept for a direct GPU-cost/accuracy comparison)")
    # HierGPS backbone
    g.add_argument("--n_pre",      type=int, default=2)
    g.add_argument("--n_gps",      type=int, default=4)
    g.add_argument("--n_pool",     type=int, default=2,   help="FPS pool levels for the token graph")
    g.add_argument("--n_inter",    type=int, default=0,   help="inter-conv layers between pool stages")
    g.add_argument("--pool_ratio", type=float, default=0.25)
    g.add_argument("--pool_knn_k", type=int, default=6)
    g.add_argument("--n_heads",    type=int, default=4)
    g.add_argument("--attn_type",  default="performer")
    g.add_argument("--grad_ckpt",  action="store_true", help="gradient checkpointing on GPS layers")
    g.add_argument("--bidirectional", action="store_true", default=True,
                   help="symmetrize the encoder's coarse-level kNN graph rebuilt after each "
                        "FPS pool (a kNN graph is inherently directed); full-resolution "
                        "data.edge_index is symmetrized upstream via the SymmetrizeEdges "
                        "dataset transform, which also fixes LocalMeshGather's k-hop base")
    g.add_argument("--no_bidirectional", dest="bidirectional", action="store_false")
    # PointNet backbone
    g.add_argument("--base_ch",    type=int, default=64)
    return parser


def build_model(args, stats, device):
    common = dict(dim_emb=args.dim_emb, dim_field=args.dim_field, n_blocks=args.n_blocks,
                  xattn_heads=args.xattn_heads, n_freqs=args.n_freqs, dim_te=args.dim_te,
                  dim_wfe=args.dim_wfe, n_geo_dist=args.n_geo_dist,
                  geometry_token_pool=args.geometry_token_pool,
                  n_geometry_tokens=args.n_geometry_tokens,
                  token_pool_heads=args.token_pool_heads,
                  token_pool_layers=args.token_pool_layers,
                  dim_out=3, query_ratio=args.query_sample_ratio,
                  query_max_points=args.query_max_points,
                  local_hops=args.local_hops, local_heads=args.local_heads,
                  local_module=args.local_module)
    if args.backbone == "hgps":
        model = build_wss_query_field(
            "hgps", n_cot=args.lap_k, dim_pe=args.dim_pe, n_pre=args.n_pre, n_gps=args.n_gps,
            n_pool=args.n_pool, n_inter=args.n_inter, pool_ratio=args.pool_ratio,
            pool_knn_k=args.pool_knn_k, n_heads=args.n_heads, attn_type=args.attn_type,
            pe_n_heads=args.pe_n_heads, pe_n_layers=args.pe_n_layers, pe_use_mlp=args.pe_use_mlp,
            grad_ckpt=args.grad_ckpt, bidirectional=args.bidirectional, **common)
    else:
        model = build_wss_query_field("pointnet", base_ch=args.base_ch, **common)
    return model.to(device)
