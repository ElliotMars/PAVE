"""Strict runtime invariants for progressive online smoke tests."""

from __future__ import annotations

from typing import Iterable

import torch


class StrictOnlineChecker:
    """Validate online tensors, lifecycle uniqueness, and update ordering."""

    def __init__(self, enabled: bool, atol: float = 1e-4) -> None:
        self.enabled = bool(enabled)
        self.atol = float(atol)
        self.reset()

    def reset(self) -> None:
        self.failure_count = 0
        self.completed_origins: set[int] = set()
        self.updated_origins: set[int] = set()
        self.committed_origins: set[int] = set()

    @staticmethod
    def _where(mask: torch.Tensor) -> tuple[int, ...]:
        indices = torch.nonzero(mask, as_tuple=False)
        return tuple(int(value) for value in indices[0].tolist())

    def _fail(
        self,
        message: str,
        origin: int,
        horizon: int = -1,
        channel: int = -1,
        expert: int = -1,
    ) -> None:
        self.failure_count += 1
        raise RuntimeError(
            "strict online check failed: {} "
            "(origin={}, horizon={}, channel={}, expert={})".format(
                message, origin, horizon, channel, expert
            )
        )

    def finite(self, name: str, tensor: torch.Tensor, origin: int) -> None:
        if not self.enabled:
            return
        invalid = ~torch.isfinite(tensor)
        if bool(invalid.any().item()):
            index = self._where(invalid)
            horizon = index[-3] if len(index) >= 3 else -1
            channel = index[-2] if len(index) >= 2 else -1
            expert = index[-1] if len(index) >= 1 else -1
            self._fail(
                f"{name} contains NaN or Inf",
                origin,
                horizon,
                channel,
                expert,
            )

    def distribution(
        self, name: str, tensor: torch.Tensor, origin: int
    ) -> None:
        if not self.enabled:
            return
        self.finite(name, tensor, origin)
        if bool((tensor < -self.atol).any().item()):
            index = self._where(tensor < -self.atol)
            self._fail(
                f"{name} contains a negative weight",
                origin,
                index[-3] if len(index) >= 3 else -1,
                index[-2] if len(index) >= 2 else -1,
                index[-1] if index else -1,
            )
        sums = tensor.sum(dim=-1)
        invalid = ~torch.isclose(
            sums,
            torch.ones_like(sums),
            atol=self.atol,
            rtol=self.atol,
        )
        if bool(invalid.any().item()):
            index = self._where(invalid)
            self._fail(
                f"{name} is not normalized",
                origin,
                index[-2] if len(index) >= 2 else -1,
                index[-1] if index else -1,
            )

    def prediction_bundle(
        self,
        prediction: torch.Tensor,
        prior: torch.Tensor,
        effective: torch.Tensor,
        correction: torch.Tensor,
        origin: int,
    ) -> None:
        if not self.enabled:
            return
        self.finite("prediction", prediction, origin)
        self.distribution("Router prior", prior, origin)
        self.distribution("effective weights", effective, origin)
        self.finite("routing correction z", correction, origin)

    def feedback(self, events: Iterable[object], origin: int) -> None:
        if not self.enabled:
            return
        for event in events:
            horizon = int(event.horizon_index)
            self.distribution(
                "local responsibility",
                event.local_responsibility,
                origin,
            )
            record = event.record
            self.distribution(
                "sample responsibility",
                record.sample_responsibility,
                origin,
            )
            self.finite(
                "local confidence", event.local_confidence, origin
            )
            unmatured = ~record.matured_mask
            if bool(torch.isfinite(record.matured_targets[unmatured]).any()):
                self._fail(
                    "unmatured target is visible", origin, horizon=horizon
                )

    def alignment(self, values: torch.Tensor, origin: int) -> None:
        if not self.enabled:
            return
        self.finite("capability alignment", values, origin)
        invalid = (values < -self.atol) | (values > 1.0 + self.atol)
        if bool(invalid.any().item()):
            expert = self._where(invalid)[-1]
            self._fail(
                "capability alignment is outside [0,1]",
                origin,
                expert=expert,
            )

    def new_record(self, record: object, origin: int) -> None:
        if not self.enabled:
            return
        if record.num_matured != 0 or bool(record.matured_mask.any().item()):
            self._fail("new record is already mature", origin)
        if not bool(torch.isnan(record.matured_targets).all().item()):
            self._fail("new record contains a future target", origin)
        forbidden = {"target", "future_target", "true", "batch_y"}
        if forbidden.intersection(record.metadata):
            self._fail("record metadata contains a future target", origin)

    def begin_completed(self, record_origin: int, origin: int) -> None:
        if not self.enabled:
            return
        if record_origin in self.completed_origins:
            self._fail("full record completed twice", origin)
        self.completed_origins.add(record_origin)

    def expert_updated(self, record_origin: int, origin: int) -> None:
        if not self.enabled:
            return
        if record_origin in self.updated_origins:
            self._fail("full record updated Expert twice", origin)
        self.updated_origins.add(record_origin)

    def before_memory_commit(self, record_origin: int, origin: int) -> None:
        if not self.enabled:
            return
        if record_origin not in self.updated_origins:
            self._fail("memory commit happened before Expert update", origin)
        if record_origin in self.committed_origins:
            self._fail("full record committed twice", origin)
        self.committed_origins.add(record_origin)

    def memory(self, manager: object, origin: int) -> None:
        if not self.enabled:
            return
        for expert_id, (stable, recovery) in enumerate(
            zip(manager.stable_buffers, manager.recovery_buffers)
        ):
            stable_ids = [item.sample_id for item in stable.items]
            recovery_ids = [item.sample_id for item in recovery.items]
            if len(stable_ids) != len(set(stable_ids)):
                self._fail(
                    "duplicate Stable sample ID", origin, expert=expert_id
                )
            if len(recovery_ids) != len(set(recovery_ids)):
                self._fail(
                    "duplicate Recovery sample ID", origin, expert=expert_id
                )
            if set(stable_ids).intersection(recovery_ids):
                self._fail(
                    "sample ID exists in Stable and Recovery",
                    origin,
                    expert=expert_id,
                )

    def subspaces(self, protector: object, origin: int) -> None:
        if not self.enabled:
            return
        for expert_id, state in enumerate(protector.states):
            basis = state.basis
            self.finite("subspace basis", basis, origin)
            if basis.numel() == 0:
                continue
            gram = basis.T @ basis
            identity = torch.eye(
                gram.shape[0], dtype=gram.dtype, device=gram.device
            )
            if not torch.allclose(
                gram, identity, atol=10 * self.atol, rtol=10 * self.atol
            ):
                self._fail(
                    "subspace basis is not orthogonal",
                    origin,
                    expert=expert_id,
                )

    def gradients(
        self,
        parameters: Iterable[torch.nn.Parameter],
        origin: int,
        label: str,
    ) -> None:
        if not self.enabled:
            return
        for expert_id, parameter in enumerate(parameters):
            if parameter.grad is None:
                continue
            if not bool(torch.isfinite(parameter.grad).all().item()):
                self._fail(
                    f"{label} gradient contains NaN or Inf",
                    origin,
                    expert=expert_id,
                )
