from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LoadProfile:
    name: str
    max_verify_tokens: int
    min_survival_probability: float

    def __post_init__(self) -> None:
        if self.max_verify_tokens < 0:
            raise ValueError("max_verify_tokens 不能小于 0")
        if not 0 <= self.min_survival_probability <= 1:
            raise ValueError("min_survival_probability 必须在 [0, 1] 内")


@dataclass(frozen=True)
class ScheduledPrefix:
    length: int
    survival_probabilities: list[float]
    stop_reason: str = "exhausted"
    rows: list["ScheduleRow"] | None = None


@dataclass(frozen=True)
class ScheduleRow:
    position: int
    confidence: float
    survival_probability: float
    status: str


LOAD_PROFILES = {
    "low": LoadProfile("low", max_verify_tokens=16, min_survival_probability=0.12),
    "medium": LoadProfile("medium", max_verify_tokens=8, min_survival_probability=0.25),
    "high": LoadProfile("high", max_verify_tokens=4, min_survival_probability=0.45),
}


def resolve_load(load: str | LoadProfile) -> LoadProfile:
    if isinstance(load, LoadProfile):
        return load
    if load not in LOAD_PROFILES:
        known = ", ".join(sorted(LOAD_PROFILES))
        raise ValueError(f"未知 load: {load}，可选值: {known}")
    return LOAD_PROFILES[load]


def schedule_prefix(confidences: list[float], load: str | LoadProfile = "medium") -> ScheduledPrefix:
    profile = resolve_load(load)
    survival_probabilities: list[float] = []
    rows: list[ScheduleRow] = []
    survival = 1.0
    stop_reason = "exhausted"

    for position, confidence in enumerate(confidences):
        if len(survival_probabilities) >= profile.max_verify_tokens:
            stop_reason = "max_verify_tokens"
            rows.append(
                ScheduleRow(
                    position=position,
                    confidence=confidence,
                    survival_probability=round(survival, 4),
                    status="pruned_by_load",
                )
            )
            break
        if not 0 <= confidence <= 1:
            raise ValueError("confidence 必须在 [0, 1] 内")

        candidate_survival = survival * confidence
        if candidate_survival < profile.min_survival_probability:
            stop_reason = "min_survival_probability"
            rows.append(
                ScheduleRow(
                    position=position,
                    confidence=confidence,
                    survival_probability=round(candidate_survival, 4),
                    status="pruned_by_confidence",
                )
            )
            break

        survival = candidate_survival
        rounded_survival = round(survival, 4)
        survival_probabilities.append(rounded_survival)
        rows.append(
            ScheduleRow(
                position=position,
                confidence=confidence,
                survival_probability=rounded_survival,
                status="scheduled",
            )
        )

    if profile.max_verify_tokens == 0:
        stop_reason = "max_verify_tokens"

    return ScheduledPrefix(
        length=len(survival_probabilities),
        survival_probabilities=survival_probabilities,
        stop_reason=stop_reason,
        rows=rows,
    )
