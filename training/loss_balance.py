"""Adaptive weighting for an auxiliary loss against a primary one.

Balances by comparing gradient norms at a shared intermediate tensor (e.g. the
model's predicted field) rather than at the full network parameters: d(loss)/d(tensor)
is one hop from a cheap analytic loss (huber/mse, or a local graph-stencil scatter),
so sizing the weight costs two small `autograd.grad` calls, not an extra backward
pass through the backbone.
"""
from __future__ import annotations

import torch


class GradNormBalancer:
    """Rescales an auxiliary loss weight so its gradient at `wrt` stays a target
    fraction of a primary loss's gradient there, tracked with an EMA so one noisy
    batch can't swing it.

    Call `update` only occasionally (e.g. once every few epochs), not every step —
    it's cheap, but there is no need to rebalance faster than the loss scales drift.

    Usage:
        balancer = GradNormBalancer(target_ratio=0.3, init_lambda=args.lambda_div)
        ...
        lam_aux = balancer.update(loss_primary, loss_aux, wrt=vel_pred)
    """

    def __init__(self, target_ratio=0.3, ema_decay=0.9, min_lambda=1e-4, max_lambda=10.0,
                init_lambda=1.0):
        self.target_ratio = target_ratio
        self.ema_decay = ema_decay
        self.min_lambda = min_lambda
        self.max_lambda = max_lambda
        self.lam = init_lambda

    @staticmethod
    def _grad_norm(loss, wrt):
        grad, = torch.autograd.grad(loss, wrt, retain_graph=True, allow_unused=True)
        return 0.0 if grad is None else grad.detach().norm().item()

    def update(self, loss_primary, loss_aux, wrt):
        """`loss_primary`/`loss_aux` are unweighted scalar loss tensors, both still
        attached to the live graph (call before `.backward()`). Returns the updated
        auxiliary weight as a plain float."""
        g_primary = self._grad_norm(loss_primary, wrt)
        g_aux = self._grad_norm(loss_aux, wrt)
        if g_aux > 0:
            target = self.target_ratio * g_primary / g_aux
            target = min(max(target, self.min_lambda), self.max_lambda)
            self.lam = self.ema_decay * self.lam + (1 - self.ema_decay) * target
        return self.lam
