#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch",
#   "torchvision",
# ]
# ///
"""
Clean checkpoint script for the Fashion-MNIST Adam-metric curvature experiment.

This intentionally removes the dead branches from the exploratory v14-v22 scripts.
It keeps only the mechanisms that survived the experiments:

  mode=adam
    Plain custom Adam on the selected objective.

  mode=adam_curv
    Adam base step plus a bounded, fresh-gated, coherent low-rank curvature
    correction.  The correction basis and correction coefficients are built
    from the same accumulated curvature batches.

Models:
  cnn
    Small Fashion-MNIST convolutional classifier.

  tiny_vit
    Tiny ViT-style image classifier.  Patch embedding, CLS token,
    learned positional embedding, Transformer blocks, and a linear head.

Objectives:
  mse
    One-hot least-squares classifier loss:
        0.5 * mean((logits - one_hot(y))^2)
    Curvature is ordinary Gauss-Newton for this squared residual.

  ce
    Standard cross entropy on logits.
    Curvature correction uses the generalized Gauss-Newton/Fisher curvature
    of the softmax-cross-entropy loss:
        J_logits^T (diag(p) - p p^T) J_logits / batch_size

The script is deliberately full-model rather than tiled.  The v22 winning
configuration used tile_size=6 for a six-block model, so this is the same
research path without the abandoned tile/sketch/batch branches.
"""

import argparse
import csv
import json
import math
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.func import functional_call
except Exception:  # pragma: no cover - older torch fallback
    from torch.nn.utils.stateless import functional_call


# -----------------------------
# Model
# -----------------------------


class FashionMNISTConvClassifier(nn.Module):
    def __init__(self, channels: int = 16, num_classes: int = 10):
        super().__init__()
        c = channels
        self.blocks = nn.ModuleList([
            nn.Conv2d(1, c, kernel_size=3, padding=1),                  # 28 -> 28
            nn.Conv2d(c, 2 * c, kernel_size=3, stride=2, padding=1),     # 28 -> 14
            nn.Conv2d(2 * c, 2 * c, kernel_size=3, stride=2, padding=1), # 14 -> 7
            nn.Conv2d(2 * c, 4 * c, kernel_size=3, stride=2, padding=1), # 7 -> 4
            nn.Linear(4 * c * 4 * 4, 4 * c),
            nn.Linear(4 * c, num_classes),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.blocks[0](x))
        h = F.gelu(self.blocks[1](h))
        h = F.gelu(self.blocks[2](h))
        h = F.gelu(self.blocks[3](h))
        h = h.flatten(1)
        h = F.gelu(self.blocks[4](h))
        return self.blocks[5](h)

    def forward_with_flat(self, x: torch.Tensor, flat: torch.Tensor) -> torch.Tensor:
        offset = 0

        def apply_layer(i: int, h: torch.Tensor) -> torch.Tensor:
            nonlocal offset
            layer = self.blocks[i]
            w_numel = layer.weight.numel()
            b_numel = 0 if layer.bias is None else layer.bias.numel()
            w = flat[offset: offset + w_numel].view_as(layer.weight)
            offset += w_numel
            b = None
            if layer.bias is not None:
                b = flat[offset: offset + b_numel].view_as(layer.bias)
                offset += b_numel

            if isinstance(layer, nn.Conv2d):
                return F.conv2d(
                    h, w, b,
                    stride=layer.stride,
                    padding=layer.padding,
                    dilation=layer.dilation,
                    groups=layer.groups,
                )
            if isinstance(layer, nn.Linear):
                return F.linear(h, w, b)
            raise TypeError(f"Unsupported layer type: {type(layer)}")

        h = F.gelu(apply_layer(0, x))
        h = F.gelu(apply_layer(1, h))
        h = F.gelu(apply_layer(2, h))
        h = F.gelu(apply_layer(3, h))
        h = h.flatten(1)
        h = F.gelu(apply_layer(4, h))
        out = apply_layer(5, h)

        if offset != flat.numel():
            raise RuntimeError(f"Consumed {offset} params but flat has {flat.numel()}")
        return out


class TinyViTBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_dim: int):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"vit_dim={dim} must be divisible by vit_heads={heads}")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5

        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, ntok, dim = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).view(bsz, ntok, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        h = (attn @ v).transpose(1, 2).contiguous().view(bsz, ntok, dim)
        x = x + self.proj(h)

        h = self.norm2(x)
        h = self.fc2(F.gelu(self.fc1(h)))
        return x + h


class TinyViTClassifier(nn.Module):
    def __init__(
        self,
        *,
        image_size: int = 28,
        patch_size: int = 7,
        dim: int = 32,
        depth: int = 2,
        heads: int = 4,
        mlp_dim: int = 64,
        num_classes: int = 10,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.patch_size = patch_size
        n_side = image_size // patch_size
        self.num_patches = n_side * n_side

        self.patch_embed = nn.Conv2d(1, dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, dim))
        self.blocks = nn.ModuleList([TinyViTBlock(dim, heads, mlp_dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(h.shape[0], -1, -1)
        h = torch.cat([cls, h], dim=1) + self.pos_embed
        for block in self.blocks:
            h = block(h)
        h = self.norm(h[:, 0])
        return self.head(h)


# -----------------------------
# Data
# -----------------------------


def make_fashion_mnist_data(
    *,
    n_train: int,
    n_val: int,
    device: torch.device,
    seed: int,
    data_dir: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    try:
        from torchvision.datasets import FashionMNIST
        from torchvision import transforms
    except Exception as exc:
        raise RuntimeError("fashion_mnist requires torchvision") from exc

    transform = transforms.ToTensor()
    train_ds = FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
    test_ds = FashionMNIST(root=data_dir, train=False, download=True, transform=transform)

    x_train_all = train_ds.data.float().unsqueeze(1) / 255.0
    y_train_all = train_ds.targets.long()
    x_val_all = test_ds.data.float().unsqueeze(1) / 255.0
    y_val_all = test_ds.targets.long()

    gen = torch.Generator()
    gen.manual_seed(seed)
    perm = torch.randperm(x_train_all.shape[0], generator=gen)

    n_train = min(n_train, x_train_all.shape[0])
    n_val = min(n_val, x_val_all.shape[0])
    train_idx = perm[:n_train]
    val_idx = torch.arange(n_val)

    return (
        x_train_all[train_idx].to(device),
        y_train_all[train_idx].to(device),
        x_val_all[val_idx].to(device),
        y_val_all[val_idx].to(device),
    )


def make_synthetic_cls_data(
    *,
    n_train: int,
    n_val: int,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Small image-shaped synthetic classification data for smoke tests."""
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    prototypes = torch.randn(10, 1, 28, 28, generator=gen, device=device) * 0.5

    def sample(n: int) -> tuple[torch.Tensor, torch.Tensor]:
        y = torch.randint(0, 10, (n,), generator=gen, device=device)
        x = prototypes[y] + 0.5 * torch.randn(n, 1, 28, 28, generator=gen, device=device)
        return x.clamp(-3, 3), y

    xtr, ytr = sample(n_train)
    xva, yva = sample(n_val)
    return xtr, ytr, xva, yva


# -----------------------------
# Flat parameter helpers
# -----------------------------


def get_flat(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().flatten() for p in model.parameters()])


def set_flat(model: nn.Module, flat: torch.Tensor) -> None:
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            p.copy_(flat[offset:offset + n].view_as(p))
            offset += n
    if offset != flat.numel():
        raise RuntimeError(f"Consumed {offset} params but flat has {flat.numel()}")


def flat_to_param_dict(model: nn.Module, flat: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
    params: OrderedDict[str, torch.Tensor] = OrderedDict()
    offset = 0
    for name, p in model.named_parameters():
        n = p.numel()
        params[name] = flat[offset:offset + n].view_as(p)
        offset += n
    if offset != flat.numel():
        raise RuntimeError(f"Consumed {offset} params but flat has {flat.numel()}")
    return params


def forward_with_flat(model: nn.Module, x: torch.Tensor, flat: torch.Tensor) -> torch.Tensor:
    # Using functional_call keeps JVP/VJP code model-agnostic.  Any model with
    # an ordinary forward can now use the curvature path.
    params = flat_to_param_dict(model, flat)
    return functional_call(model, params, (x,))


# -----------------------------
# Objective helpers
# -----------------------------


def loss_from_logits(logits: torch.Tensor, labels: torch.Tensor, objective: str) -> torch.Tensor:
    if objective == "ce":
        return F.cross_entropy(logits, labels, reduction="mean")
    if objective == "mse":
        target = F.one_hot(labels, num_classes=logits.shape[1]).to(dtype=logits.dtype)
        return 0.5 * (logits - target).square().mean()
    raise ValueError(f"Unknown objective: {objective}")


def loss_from_flat(
    model: nn.Module,
    flat: torch.Tensor,
    xb: torch.Tensor,
    yb: torch.Tensor,
    objective: str,
) -> torch.Tensor:
    return loss_from_logits(forward_with_flat(model, xb, flat), yb, objective)


@torch.no_grad()
def eval_metrics(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    objective: str,
    batch_size: int = 2048,
) -> tuple[float, float]:
    total_loss = 0.0
    total_correct = 0
    n = x.shape[0]
    for start in range(0, n, batch_size):
        xb = x[start:start + batch_size]
        yb = y[start:start + batch_size]
        logits = model(xb)
        total_loss += float(loss_from_logits(logits, yb, objective).item()) * xb.shape[0]
        total_correct += int((logits.argmax(dim=1) == yb).sum().item())
    return total_loss / n, total_correct / n


def choose_batch(n: int, batch_size: int, device: torch.device) -> torch.Tensor:
    if batch_size >= n:
        return torch.arange(n, device=device)
    return torch.randint(0, n, (batch_size,), device=device)


# -----------------------------
# Linear algebra utilities
# -----------------------------


def orthonormalize_columns(cols: Sequence[torch.Tensor], *, eps: float = 1e-12) -> torch.Tensor:
    basis: list[torch.Tensor] = []
    for col in cols:
        v = col.detach().clone()
        for q in basis:
            v = v - q * torch.dot(q, v)
        for q in basis:  # one reorthogonalisation pass
            v = v - q * torch.dot(q, v)
        nrm = torch.linalg.vector_norm(v)
        if torch.isfinite(nrm) and float(nrm.item()) > eps:
            basis.append(v / nrm)
    if not basis:
        device = cols[0].device
        dtype = cols[0].dtype
        return torch.empty((cols[0].numel(), 0), device=device, dtype=dtype)
    return torch.stack(basis, dim=1)


def krylov_basis(
    anchor: torch.Tensor,
    matvec: Callable[[torch.Tensor], torch.Tensor],
    rank: int,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    if rank <= 0:
        return torch.empty((anchor.numel(), 0), device=anchor.device, dtype=anchor.dtype)
    basis: list[torch.Tensor] = []
    v = anchor.detach().clone()
    for _ in range(rank):
        for q in basis:
            v = v - q * torch.dot(q, v)
        for q in basis:
            v = v - q * torch.dot(q, v)
        nrm = torch.linalg.vector_norm(v)
        if not torch.isfinite(nrm) or float(nrm.item()) <= eps:
            break
        q = v / nrm
        basis.append(q)
        v = matvec(q).detach()
    if not basis:
        return torch.empty((anchor.numel(), 0), device=anchor.device, dtype=anchor.dtype)
    return torch.stack(basis, dim=1)


def softmax_ce_hvp_logits(logits: torch.Tensor, jv_logits: torch.Tensor) -> torch.Tensor:
    p = logits.softmax(dim=1)
    inner = (p * jv_logits).sum(dim=1, keepdim=True)
    return p * (jv_logits - inner) / float(logits.shape[0])


def ggn_flat_matvec(
    *,
    model: nn.Module,
    flat0: torch.Tensor,
    xb: torch.Tensor,
    yb: torch.Tensor,
    v_phys: torch.Tensor,
    objective: str,
) -> torch.Tensor:
    """Generalized Gauss-Newton matvec in parameter space at flat0."""
    with torch.enable_grad():
        flat_for_jvp = flat0.detach().clone().requires_grad_(True)

        def logits_fn(z: torch.Tensor) -> torch.Tensor:
            return forward_with_flat(model, xb, z)

        logits, jv_logits = torch.autograd.functional.jvp(
            logits_fn, flat_for_jvp, v_phys.detach(), create_graph=False, strict=False
        )

        flat_for_vjp = flat0.detach().clone().requires_grad_(True)
        logits_vjp = forward_with_flat(model, xb, flat_for_vjp)
        if objective == "mse":
            # Hessian wrt logits for 0.5 * mean((logits - one_hot)^2) is I / numel.
            hv_logits = jv_logits.detach() / float(logits_vjp.numel())
        elif objective == "ce":
            # GGN/Fisher Hessian wrt logits for mean softmax cross entropy.
            hv_logits = softmax_ce_hvp_logits(logits.detach(), jv_logits.detach())
        else:
            raise ValueError(f"Unknown objective: {objective}")

        hv_flat = torch.autograd.grad(
            logits_vjp, flat_for_vjp, grad_outputs=hv_logits, create_graph=False
        )[0]
    return hv_flat.detach()


def objective_grad_flat(
    *,
    model: nn.Module,
    flat0: torch.Tensor,
    xb: torch.Tensor,
    yb: torch.Tensor,
    objective: str,
) -> tuple[torch.Tensor, float]:
    flat_req = flat0.detach().clone().requires_grad_(True)
    loss = loss_from_flat(model, flat_req, xb, yb, objective)
    g = torch.autograd.grad(loss, flat_req, create_graph=False)[0].detach()
    return g, float(loss.detach().item())


# -----------------------------
# Optimizer step
# -----------------------------


@dataclass
class StepStats:
    step_loss: float
    accepted: int
    gate_advantage: float
    correction_norm_frac: float
    accum_batches_used: int
    nominal_curv_matvecs: int


def adam_curvature_step(
    *,
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    objective: str,
    batch_size: int,
    lr: float,
    damping: float,
    beta1: float,
    beta2: float,
    eps: float,
    m: torch.Tensor,
    v: torch.Tensor,
    t: int,
    rank: int,
    damping_mult: float,
    max_norm_frac: float,
    accum_batches: int,
    accum_batch_size: int,
    include_base_batch: bool,
    acceptance_mode: str,
    acceptance_batch_size: int,
    acceptance_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, StepStats]:
    device = x_train.device
    n_train = x_train.shape[0]
    flat0 = get_flat(model).to(device)

    base_idx = choose_batch(n_train, batch_size, device)
    xb = x_train[base_idx]
    yb = y_train[base_idx]

    g_current, step_loss = objective_grad_flat(
        model=model, flat0=flat0, xb=xb, yb=yb, objective=objective
    )

    m_new = beta1 * m + (1.0 - beta1) * g_current
    v_new = beta2 * v + (1.0 - beta2) * g_current.square()
    m_hat = m_new / (1.0 - beta1 ** t)
    v_hat = v_new / (1.0 - beta2 ** t)

    metric = v_hat.sqrt().add(eps).detach()
    inv_sqrt_metric = metric.rsqrt()

    # Match v22 convention: damping=300 gives effective Adam lr 1/300.
    # If lr is supplied positive, it overrides this effective LR.
    effective_lr = float(lr) if float(lr) > 0 else 1.0 / max(float(damping), 1e-30)
    rhs_white = inv_sqrt_metric * m_hat.detach()
    z_base = -effective_lr * rhs_white
    delta_base = inv_sqrt_metric * z_base

    # Coherent curvature batches: the same list defines Krylov directions,
    # correction coefficients, and H * delta_base.
    corr_batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    if include_base_batch:
        corr_batches.append((xb, yb))
    extra_needed = max(0, int(accum_batches) - len(corr_batches))
    b_corr = int(accum_batch_size) if int(accum_batch_size) > 0 else int(batch_size)
    for _ in range(extra_needed):
        idx = choose_batch(n_train, b_corr, device)
        corr_batches.append((x_train[idx], y_train[idx]))
    if not corr_batches:
        corr_batches.append((xb, yb))

    curv_calls = 0

    def white_curv_matvec(s_white: torch.Tensor) -> torch.Tensor:
        nonlocal curv_calls
        curv_calls += 1
        v_phys = inv_sqrt_metric * s_white.detach()
        total = torch.zeros_like(s_white)
        for xb_i, yb_i in corr_batches:
            hv_phys = ggn_flat_matvec(
                model=model, flat0=flat0, xb=xb_i, yb=yb_i,
                v_phys=v_phys, objective=objective,
            )
            total.add_(inv_sqrt_metric * hv_phys)
        return total / float(len(corr_batches))

    q_corr = max(0, int(rank))
    accepted = 0
    gate_advantage = 0.0
    correction_norm_frac = 0.0

    if q_corr <= 0:
        delta = delta_base
    else:
        # Build [a, Ka, K^2 a, ...] and drop the Adam axis.
        basis_full = krylov_basis(rhs_white.detach(), white_curv_matvec, q_corr + 1)
        S_corr = basis_full[:, 1:] if basis_full.shape[1] > 1 else torch.empty((flat0.numel(), 0), device=device, dtype=flat0.dtype)

        if S_corr.numel() == 0 or S_corr.shape[1] == 0:
            delta = delta_base
        else:
            q = S_corr.shape[1]
            KS = torch.stack([white_curv_matvec(S_corr[:, j]) for j in range(q)], dim=1)
            Kz_base = white_curv_matvec(z_base)

            corr_damping = float(damping) * max(float(damping_mult), 0.0)
            H_small = S_corr.T @ KS + corr_damping * (S_corr.T @ S_corr)
            H_small = 0.5 * (H_small + H_small.T)
            jitter = 1e-10 * H_small.diag().abs().mean().clamp_min(1.0)
            H_small = H_small + jitter * torch.eye(q, device=device, dtype=H_small.dtype)

            rhs_small = S_corr.T @ (rhs_white + Kz_base)
            try:
                y_corr = torch.linalg.solve(H_small, -rhs_small)
            except RuntimeError:
                y_corr = torch.linalg.lstsq(H_small, -rhs_small).solution

            z_corr = S_corr @ y_corr
            base_norm = torch.linalg.vector_norm(z_base).clamp_min(1e-30)
            corr_norm = torch.linalg.vector_norm(z_corr)
            max_corr = float(max_norm_frac) * base_norm
            if float(corr_norm.item()) > float(max_corr.item()):
                z_corr = z_corr * (max_corr / corr_norm)
                corr_norm = max_corr
            correction_norm_frac = float((corr_norm / base_norm).item())
            delta_candidate = delta_base + inv_sqrt_metric * z_corr

            # Fresh gate compares Adam-only base step versus base+correction.
            if acceptance_mode == "same":
                xg, yg = xb, yb
            elif acceptance_mode == "fresh_train":
                gi = choose_batch(n_train, acceptance_batch_size or batch_size, device)
                xg, yg = x_train[gi], y_train[gi]
            elif acceptance_mode == "fresh_val":
                gi = choose_batch(x_val.shape[0], acceptance_batch_size or batch_size, device)
                xg, yg = x_val[gi], y_val[gi]
            else:
                raise ValueError(f"Unknown acceptance_mode: {acceptance_mode}")

            with torch.no_grad():
                loss_base = float(loss_from_flat(model, flat0 + delta_base, xg, yg, objective).item())
                loss_corr = float(loss_from_flat(model, flat0 + delta_candidate, xg, yg, objective).item())
            gate_advantage = loss_base - loss_corr
            if gate_advantage > float(acceptance_threshold):
                delta = delta_candidate
                accepted = 1
            else:
                delta = delta_base

    set_flat(model, flat0 + delta.detach())
    return m_new.detach(), v_new.detach(), StepStats(
        step_loss=step_loss,
        accepted=accepted,
        gate_advantage=gate_advantage,
        correction_norm_frac=correction_norm_frac,
        accum_batches_used=len(corr_batches),
        nominal_curv_matvecs=curv_calls,
    )


def adam_step(
    *,
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    objective: str,
    batch_size: int,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    m: torch.Tensor,
    v: torch.Tensor,
    t: int,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    device = x_train.device
    idx = choose_batch(x_train.shape[0], batch_size, device)
    xb = x_train[idx]
    yb = y_train[idx]
    flat0 = get_flat(model).to(device)
    g_current, loss = objective_grad_flat(model=model, flat0=flat0, xb=xb, yb=yb, objective=objective)

    m_new = beta1 * m + (1.0 - beta1) * g_current
    v_new = beta2 * v + (1.0 - beta2) * g_current.square()
    m_hat = m_new / (1.0 - beta1 ** t)
    v_hat = v_new / (1.0 - beta2 ** t)
    delta = -float(lr) * m_hat / (v_hat.sqrt() + eps)
    set_flat(model, flat0 + delta.detach())
    return m_new.detach(), v_new.detach(), loss


# -----------------------------
# Main
# -----------------------------


def make_out_name(args: argparse.Namespace) -> str:
    parts = [
        args.problem,
        args.model,
        args.objective,
        args.mode,
        f"seed{args.seed}",
        f"steps{args.steps}",
        f"bs{args.batch_size}",
    ]
    if args.model == "cnn":
        parts.append(f"conv{args.conv_channels}")
    else:
        parts.extend([
            f"vitp{args.vit_patch_size}",
            f"d{args.vit_dim}",
            f"L{args.vit_depth}",
            f"h{args.vit_heads}",
            f"mlp{args.vit_mlp_dim}",
        ])
    if args.mode == "adam":
        parts.append(f"lr{args.lr:g}")
    else:
        parts.extend([
            f"damp{args.damping:g}",
            f"r{args.correction_rank}",
            f"dm{args.correction_damping_mult:g}",
            f"cap{args.correction_max_norm_frac:g}",
            f"acc{args.correction_accum_batches}",
            f"gate{args.acceptance_mode}",
        ])
    return "_".join(parts)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clean Adam + coherent curvature correction experiment")
    p.add_argument("--problem", choices=["fashion_mnist", "synthetic_cls"], default="fashion_mnist")
    p.add_argument("--model", choices=["cnn", "tiny_vit"], default="cnn")
    p.add_argument("--objective", choices=["mse", "ce"], default="mse")
    p.add_argument("--mode", choices=["adam", "adam_curv"], default="adam_curv")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default="runs")
    p.add_argument("--data_dir", default="./data")

    p.add_argument("--n_train", type=int, default=8192)
    p.add_argument("--n_val", type=int, default=2048)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--conv_channels", type=int, default=16)
    p.add_argument("--vit_patch_size", type=int, default=7)
    p.add_argument("--vit_dim", type=int, default=32)
    p.add_argument("--vit_depth", type=int, default=2)
    p.add_argument("--vit_heads", type=int, default=4)
    p.add_argument("--vit_mlp_dim", type=int, default=64)

    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--eval_every", type=int, default=50)
    p.add_argument("--eval_batch_size", type=int, default=2048)

    p.add_argument("--lr", type=float, default=0.003333333333333333)
    p.add_argument("--damping", type=float, default=300.0, help="Used as lr=1/damping when --lr_for_curv <= 0")
    p.add_argument("--lr_for_curv", type=float, default=0.0, help="Override Adam base lr in adam_curv; 0 means 1/damping")
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--eps", type=float, default=1e-8)

    p.add_argument("--correction_rank", type=int, default=2)
    p.add_argument("--correction_damping_mult", type=float, default=10.0)
    p.add_argument("--correction_max_norm_frac", type=float, default=0.25)
    p.add_argument("--correction_accum_batches", type=int, default=2)
    p.add_argument("--correction_accum_batch_size", type=int, default=64)
    p.add_argument("--correction_include_base_batch", type=int, default=1)

    p.add_argument("--acceptance_mode", choices=["same", "fresh_train", "fresh_val"], default="fresh_train")
    p.add_argument("--acceptance_batch_size", type=int, default=256)
    p.add_argument("--acceptance_threshold", type=float, default=0.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)

    if args.problem == "fashion_mnist":
        x_train, y_train, x_val, y_val = make_fashion_mnist_data(
            n_train=args.n_train, n_val=args.n_val, device=device, seed=args.seed, data_dir=args.data_dir
        )
    else:
        x_train, y_train, x_val, y_val = make_synthetic_cls_data(
            n_train=args.n_train, n_val=args.n_val, device=device, seed=args.seed
        )

    if args.model == "cnn":
        model = FashionMNISTConvClassifier(channels=args.conv_channels).to(device)
    elif args.model == "tiny_vit":
        model = TinyViTClassifier(
            patch_size=args.vit_patch_size,
            dim=args.vit_dim,
            depth=args.vit_depth,
            heads=args.vit_heads,
            mlp_dim=args.vit_mlp_dim,
        ).to(device)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    flat0 = get_flat(model).to(device)
    m = torch.zeros_like(flat0)
    v = torch.zeros_like(flat0)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_name = make_out_name(args)
    csv_path = out_dir / f"{base_name}.csv"
    args_path = out_dir / f"{base_name}.args.json"

    fieldnames = [
        "step", "mode", "objective", "train_loss", "val_loss", "train_acc", "val_acc",
        "batch_loss", "best_val_loss", "best_val_step", "accepted", "gate_advantage",
        "correction_norm_frac", "accum_batches_used", "nominal_curv_matvecs", "elapsed_sec",
    ]

    best_val = float("inf")
    best_step = 0
    start_time = time.time()

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for step in range(1, args.steps + 1):
            if args.mode == "adam":
                m, v, batch_loss = adam_step(
                    model=model, x_train=x_train, y_train=y_train, objective=args.objective,
                    batch_size=args.batch_size, lr=args.lr, beta1=args.beta1, beta2=args.beta2,
                    eps=args.eps, m=m, v=v, t=step,
                )
                stats = StepStats(batch_loss, 0, 0.0, 0.0, 0, 0)
            else:
                m, v, stats = adam_curvature_step(
                    model=model, x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val,
                    objective=args.objective, batch_size=args.batch_size,
                    lr=args.lr_for_curv, damping=args.damping, beta1=args.beta1, beta2=args.beta2,
                    eps=args.eps, m=m, v=v, t=step, rank=args.correction_rank,
                    damping_mult=args.correction_damping_mult,
                    max_norm_frac=args.correction_max_norm_frac,
                    accum_batches=args.correction_accum_batches,
                    accum_batch_size=args.correction_accum_batch_size,
                    include_base_batch=bool(args.correction_include_base_batch),
                    acceptance_mode=args.acceptance_mode,
                    acceptance_batch_size=args.acceptance_batch_size,
                    acceptance_threshold=args.acceptance_threshold,
                )
                batch_loss = stats.step_loss

            if step == 1 or step % args.eval_every == 0 or step == args.steps:
                train_loss, train_acc = eval_metrics(
                    model, x_train, y_train, objective=args.objective, batch_size=args.eval_batch_size
                )
                val_loss, val_acc = eval_metrics(
                    model, x_val, y_val, objective=args.objective, batch_size=args.eval_batch_size
                )
                if val_loss < best_val:
                    best_val = val_loss
                    best_step = step
                row = {
                    "step": step,
                    "mode": args.mode,
                    "objective": args.objective,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                    "batch_loss": batch_loss,
                    "best_val_loss": best_val,
                    "best_val_step": best_step,
                    "accepted": stats.accepted,
                    "gate_advantage": stats.gate_advantage,
                    "correction_norm_frac": stats.correction_norm_frac,
                    "accum_batches_used": stats.accum_batches_used,
                    "nominal_curv_matvecs": stats.nominal_curv_matvecs,
                    "elapsed_sec": time.time() - start_time,
                }
                writer.writerow(row)
                f.flush()
                print(
                    f"step={step:05d} mode={args.mode:<9} obj={args.objective:<3} "
                    f"train={train_loss:.6g} val={val_loss:.6g} best={best_val:.6g} "
                    f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} "
                    f"acc={stats.accepted} corr_frac={stats.correction_norm_frac:.3g}"
                )

    args_path.write_text(json.dumps(vars(args), indent=2, sort_keys=True))
    print(f"\nWrote log: {csv_path}")
    print(f"Wrote args: {args_path}")
    print(f"Best val loss: {best_val} at step {best_step}")


if __name__ == "__main__":
    main()
