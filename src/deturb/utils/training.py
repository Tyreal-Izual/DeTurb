"""Testable FP32 training and validation steps for DeTurb."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from deturb.model import (
    DeTurbContract,
    DeTurbLoss,
    DeTurbOutput,
    dataset_clip_to_model_input,
    select_output_target,
)
from .numerics import require_finite, require_finite_state, tensors_in
from .checkpoints import capture_scheduler_state


@dataclass(frozen=True, slots=True)
class DeTurbStepResult:
    restored: torch.Tensor
    target: torch.Tensor
    losses: dict[str, float | bool]
    optimizer_step_completed: bool


def train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    criterion: DeTurbLoss,
    input_sequence: torch.Tensor,
    target_sequence: torch.Tensor,
    contract: DeTurbContract,
    *,
    iteration: int,
) -> DeTurbStepResult:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    require_finite((input_sequence, target_sequence), "training input", input_sequence.device, synchronize=True)
    model_input = dataset_clip_to_model_input(input_sequence, contract)
    target = select_output_target(target_sequence, contract)
    output = model(model_input, return_aux=True)
    if not isinstance(output, DeTurbOutput):
        raise TypeError("DeTurb model did not return auxiliary output")
    breakdown = criterion(output, target, iteration=iteration)
    require_finite((output.restored, breakdown.total), "training forward/loss", input_sequence.device, synchronize=True)
    breakdown.total.backward()
    try:
        require_finite(
            (p.grad for p in model.parameters() if p.grad is not None),
            "training gradients", input_sequence.device, synchronize=True,
        )
    except FloatingPointError:
        optimizer.zero_grad(set_to_none=True)
        raise
    optimizer.step()
    # Finite gradients can still overflow optimizer accumulators. Stop before
    # advancing the scheduler or writing a checkpoint if the update failed.
    require_finite(
        (*model.parameters(), *tensors_in(optimizer.state)),
        "optimizer update", input_sequence.device, synchronize=True,
    )
    if scheduler is not None:
        scheduler.step()
    policies = [{key: value for key, value in group.items() if key != "params"}
                for group in optimizer.param_groups]
    require_finite_state((policies, capture_scheduler_state(scheduler)),
                         "scheduler update", input_sequence.device, synchronize=True)
    return DeTurbStepResult(
        restored=output.restored.detach(),
        target=target.detach(),
        losses=breakdown.detached_scalars(),
        optimizer_step_completed=True,
    )


def validation_step(
    model: torch.nn.Module,
    criterion: DeTurbLoss,
    input_sequence: torch.Tensor,
    target_sequence: torch.Tensor,
    contract: DeTurbContract,
    *,
    iteration: int,
) -> DeTurbStepResult:
    model.eval()
    # Validation shards may have unequal lengths: checks here MUST be local.
    # The trainer coordinates failure once all ranks finish their shard.
    require_finite((input_sequence, target_sequence), "validation input", input_sequence.device)
    model_input = dataset_clip_to_model_input(input_sequence, contract)
    target = select_output_target(target_sequence, contract)
    with torch.inference_mode():
        output = model(model_input, return_aux=True)
        if not isinstance(output, DeTurbOutput):
            raise TypeError("DeTurb model did not return auxiliary output")
        breakdown = criterion(output, target, iteration=iteration)
        require_finite((output.restored, breakdown.total), "validation forward/loss", input_sequence.device)
    return DeTurbStepResult(
        restored=output.restored,
        target=target,
        losses=breakdown.detached_scalars(),
        optimizer_step_completed=False,
    )
