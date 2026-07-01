from __future__ import annotations

from dataclasses import dataclass

from dspark_nano.scheduler import ScheduleRow


@dataclass(frozen=True)
class StepTrace:
    mode: str
    load: str
    round_index: int
    prompt_tokens: list[int]
    draft_tokens: list[int]
    target_tokens: list[int]
    confidences: list[float]
    survival_probabilities: list[float]
    scheduled_length: int
    accepted_length: int
    scheduler_stop_reason: str
    token_rows: list[ScheduleRow]
    emitted_tokens: list[int]
    accepted_draft_tokens: list[int]
    corrected_target_tokens: list[int]
    pruned_tokens: list[int]

    @property
    def verification_waste(self) -> int:
        return self.scheduled_length - self.accepted_length

    @property
    def verification_cost(self) -> int:
        if self.scheduled_length == 0 and self.corrected_target_tokens:
            return 1
        return self.scheduled_length

    @property
    def mismatch_index(self) -> int | None:
        if self.corrected_target_tokens and self.scheduled_length > 0:
            return self.accepted_length
        return None

    def as_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "load": self.load,
            "round_index": self.round_index,
            "prompt_tokens": self.prompt_tokens,
            "draft_tokens": self.draft_tokens,
            "target_tokens": self.target_tokens,
            "confidences": self.confidences,
            "survival_probabilities": self.survival_probabilities,
            "scheduled_length": self.scheduled_length,
            "accepted_length": self.accepted_length,
            "scheduler_stop_reason": self.scheduler_stop_reason,
            "mismatch_index": self.mismatch_index,
            "emitted_tokens": self.emitted_tokens,
            "accepted_draft_tokens": self.accepted_draft_tokens,
            "corrected_target_tokens": self.corrected_target_tokens,
            "pruned_tokens": self.pruned_tokens,
            "verification_cost": self.verification_cost,
            "verification_waste": self.verification_waste,
        }

    def table(self) -> str:
        lines = [
            "pos | draft | target | conf | survival | status",
            "----|-------|--------|------|----------|--------",
        ]
        row_by_position = {row.position: row for row in self.token_rows}
        for position, (draft_token, target_token, confidence) in enumerate(
            zip(self.draft_tokens, self.target_tokens, self.confidences)
        ):
            row = row_by_position.get(position)
            survival = "-" if row is None else f"{row.survival_probability:.4f}"
            if position < self.accepted_length:
                status = "accepted"
            elif self.mismatch_index == position:
                status = "corrected"
            elif row and row.status.startswith("pruned"):
                status = row.status
            else:
                status = "not_scheduled"
            lines.append(
                f"{position:>3} | {draft_token:>5} | {target_token:>6} | {confidence:.2f} | {survival:>8} | {status}"
            )
        return "\n".join(lines)
