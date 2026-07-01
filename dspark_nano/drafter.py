from __future__ import annotations

from dataclasses import dataclass

from dspark_nano.target import ToyTarget


@dataclass(frozen=True)
class DraftBlock:
    draft_tokens: list[int]
    target_tokens: list[int]
    confidences: list[float]


class ToyProfileDrafter:
    """基于 oracle acceptance profile 的 drafter 教学模拟。

    这个类会读取 toy target token，再按手写 profile 决定哪些位置命中。
    它用于可视化 suffix decay，不是真实 draft model，也不复现 DeepSpec。
    `parallel` 模式刻意让后缀置信度快速下降；`dspark` 模式模拟 DSpark-style
    lightweight sequential head 对后缀一致性的修正。
    """

    _PARALLEL_PROFILE = [0.93, 0.78, 0.61, 0.45, 0.33, 0.24, 0.18, 0.14]
    _DSPARK_PROFILE = [0.93, 0.84, 0.76, 0.68, 0.61, 0.55, 0.50, 0.46]

    def __init__(self, mode: str = "parallel", gamma: int = 5, mismatch_cutoff: float = 0.70) -> None:
        if mode not in {"parallel", "dspark"}:
            raise ValueError("mode 只能是 'parallel' 或 'dspark'")
        if gamma <= 0:
            raise ValueError("gamma 必须大于 0")
        self.mode = mode
        self.gamma = gamma
        self.mismatch_cutoff = mismatch_cutoff

    def conditional_acceptance(self) -> list[float]:
        profile = self._DSPARK_PROFILE if self.mode == "dspark" else self._PARALLEL_PROFILE
        if self.gamma <= len(profile):
            return profile[: self.gamma]

        values = profile[:]
        tail = profile[-1]
        while len(values) < self.gamma:
            decay = 0.92 if self.mode == "dspark" else 0.75
            tail = max(0.05, round(tail * decay, 4))
            values.append(tail)
        return values

    def draft(self, prompt_tokens: list[int], target: ToyTarget) -> DraftBlock:
        target_tokens = target.next_tokens(prompt_tokens, self.gamma)
        confidences = self.conditional_acceptance()
        draft_tokens: list[int] = []

        for token, confidence in zip(target_tokens, confidences):
            if confidence >= self.mismatch_cutoff:
                draft_tokens.append(token)
            else:
                draft_tokens.append((token + 1) % target.vocab_size)

        return DraftBlock(
            draft_tokens=draft_tokens,
            target_tokens=target_tokens,
            confidences=confidences,
        )


ParallelDrafter = ToyProfileDrafter
