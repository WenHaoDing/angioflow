"""Run training/eval_joint.py with a guard around every CUDA kNN the model makes.

eval_joint.py on the flow-split joint run crashed reproducibly at the 20th test case
(aneug_flow/stable_321) with a CUDA "index out of bounds" (first at
models/joint/decoder.py:81, LocalFieldGather indexing with torch_cluster.knn's output).
Inputs are clean (no NaN/inf/duplicates) and peak memory is only 4.5 GB, but inside the
run torch_cluster's CUDA knn returned a handful of out-of-range indices (13 of 1,307,440
for that case's interior->wall search) -- a torch_cluster CUDA bug, not the data or the
model. The same calls on the CPU are correct.

Range-checking the CUDA result and recomputing bad calls on the CPU was NOT enough: the
run still crashed later in the same forward (a GATv2 index_select on indices that had
passed the check), i.e. the CUDA knn kernel writes out of bounds and corrupts other
tensors -- its bad output is only the visible part. So the CUDA kernel must not run.

This wrapper changes nothing in eval_joint.py or the model. It replaces torch_cluster.knn
and torch_cluster.knn_graph -- which every PyG knn / knn_graph / knn_interpolate call in
the model goes through (PyG looks them up at call time) -- with the CPU kernels (exact;
results moved back to the GPU); everything else still runs on the GPU. Results equal a
correct run. Same CLI as eval_joint.py
(run with PYTHONPATH = the folder that contains angioflow/):

    PYTHONPATH=<parent of angioflow> python debugging/run_eval_joint_checked.py \\
        --ckpt_root <root> --run_name <run> --which best --device cuda:0 --no_viz --full_test_set
"""
import functools

import torch
import torch_geometric        # noqa: F401 -- import FIRST: torch_geometric.typing inspects
import torch_geometric.nn     # noqa: F401    torch_cluster.knn (its docstring) to decide
import torch_cluster          #               WITH_TORCH_CLUSTER; patch only afterwards

_knn, _knn_graph = torch_cluster.knn, torch_cluster.knn_graph
_stats = {"calls": 0, "fallbacks": 0}


def _cpu(t):
    return t.cpu() if torch.is_tensor(t) else t


def _bad(edge, n_rows, n_cols):
    return (edge[0] < 0) | (edge[0] >= n_rows) | (edge[1] < 0) | (edge[1] >= n_cols)


@functools.wraps(_knn)
def checked_knn(x, y, k, batch_x=None, batch_y=None, *args, **kwargs):
    # always on the CPU: the CUDA kernel corrupts memory (see docstring), so it must not run
    out = _knn(x.cpu(), y.cpu(), k, _cpu(batch_x), _cpu(batch_y), *args, **kwargs)
    _stats["calls"] += 1
    bad = _bad(out, y.shape[0], x.shape[0])            # row = y (query) index, col = x index
    if bool(bad.any()):
        raise RuntimeError(f"CPU knn returned {int(bad.sum())} out-of-range indices")
    return out.to(x.device)


@functools.wraps(_knn_graph)
def checked_knn_graph(x, k, batch=None, *args, **kwargs):
    out = _knn_graph(x.cpu(), k, _cpu(batch), *args, **kwargs)
    _stats["calls"] += 1
    if bool(_bad(out, x.shape[0], x.shape[0]).any()):
        raise RuntimeError("CPU knn_graph returned out-of-range indices")
    return out.to(x.device)


torch_cluster.knn, torch_cluster.knn_graph = checked_knn, checked_knn_graph

if __name__ == "__main__":
    from angioflow.training.eval_joint import main
    main()
    print(f"[knn-guard] {_stats['calls']} kNN calls, all computed on the CPU")
