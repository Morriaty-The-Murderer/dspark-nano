from __future__ import annotations

from dataclasses import dataclass

from dspark_nano.config import SamplingConfig
from dspark_nano.drafter import ToyProfileDrafter
from dspark_nano.scheduler import LoadProfile, resolve_load, schedule_prefix
from dspark_nano.target import ToyTarget
from dspark_nano.trace import StepTrace
from dspark_nano.verifier import verify_prefix


@dataclass(frozen=True)
class DSparkNanoConfig:
    gamma: int = 5
    drafter_mode: str = "dspark"
    vocab_size: int = 97

    def __post_init__(self) -> None:
        if self.gamma <= 0:
            raise ValueError("gamma 必须大于 0")


@dataclass(frozen=True)
class GenerationMetrics:
    rounds: int
    accepted_tokens: int
    accepted_draft_tokens: int
    corrected_target_tokens: int
    verification_tokens: int
    verification_waste: int


@dataclass(frozen=True)
class GenerationResult:
    text: str
    token_ids: list[int]
    metrics: GenerationMetrics
    traces: list[StepTrace]


class DSparkNano:
    """一个可读、可跑的 DSpark-style speculative decoding 教学引擎。"""

    def __init__(self, config: DSparkNanoConfig | None = None) -> None:
        self.config = config or DSparkNanoConfig()
        self.target = ToyTarget(vocab_size=self.config.vocab_size)
        self.drafter = ToyProfileDrafter(mode=self.config.drafter_mode, gamma=self.config.gamma)

    def step_trace(self, prompt_tokens: list[int], load: str | LoadProfile = "medium", round_index: int = 0) -> StepTrace:
        draft = self.drafter.draft(prompt_tokens, self.target)
        scheduled = schedule_prefix(draft.confidences, load)
        accepted_length = verify_prefix(draft.draft_tokens, draft.target_tokens, scheduled.length)
        accepted_draft_tokens = draft.draft_tokens[:accepted_length]

        corrected_target_tokens: list[int] = []
        if scheduled.length == 0:
            corrected_target_tokens = draft.target_tokens[:1]
        elif accepted_length < scheduled.length:
            corrected_target_tokens = [draft.target_tokens[accepted_length]]

        emitted_tokens = accepted_draft_tokens + corrected_target_tokens
        pruned_tokens = draft.draft_tokens[scheduled.length :]
        load_name = resolve_load(load).name

        return StepTrace(
            mode=self.config.drafter_mode,
            load=load_name,
            round_index=round_index,
            prompt_tokens=list(prompt_tokens),
            draft_tokens=draft.draft_tokens,
            target_tokens=draft.target_tokens,
            confidences=draft.confidences,
            survival_probabilities=scheduled.survival_probabilities,
            scheduled_length=scheduled.length,
            accepted_length=accepted_length,
            scheduler_stop_reason=scheduled.stop_reason,
            token_rows=scheduled.rows or [],
            emitted_tokens=emitted_tokens,
            accepted_draft_tokens=accepted_draft_tokens,
            corrected_target_tokens=corrected_target_tokens,
            pruned_tokens=pruned_tokens,
        )

    def generate(self, prompt: str | list[int], sampling_config: SamplingConfig | None = None) -> GenerationResult:
        sampling = sampling_config or SamplingConfig()
        token_ids = self.target.encode(prompt) if isinstance(prompt, str) else list(prompt)
        prompt_len = len(token_ids)
        traces: list[StepTrace] = []
        verification_tokens = 0
        accepted_tokens = 0
        accepted_draft_tokens = 0
        corrected_target_tokens = 0

        while len(token_ids) - prompt_len < sampling.max_tokens:
            trace = self.step_trace(token_ids, sampling.load, round_index=len(traces))
            traces.append(trace)
            verification_tokens += trace.verification_cost

            remaining = sampling.max_tokens - (len(token_ids) - prompt_len)
            emitted = trace.emitted_tokens[:remaining]
            token_ids.extend(emitted)
            accepted_tokens += len(emitted)
            accepted_draft_count = min(len(trace.accepted_draft_tokens), len(emitted))
            accepted_draft_tokens += accepted_draft_count
            corrected_target_tokens += max(0, len(emitted) - accepted_draft_count)

        metrics = GenerationMetrics(
            rounds=len(traces),
            accepted_tokens=accepted_tokens,
            accepted_draft_tokens=accepted_draft_tokens,
            corrected_target_tokens=corrected_target_tokens,
            verification_tokens=verification_tokens,
            verification_waste=sum(trace.verification_waste for trace in traces),
        )
        generated = token_ids[prompt_len:]
        return GenerationResult(
            text=self.target.decode(generated),
            token_ids=generated,
            metrics=metrics,
            traces=traces,
        )
