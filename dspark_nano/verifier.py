from __future__ import annotations


def acceptance_length(draft_tokens: list[int], target_tokens: list[int]) -> int:
    accepted = 0
    for draft_token, target_token in zip(draft_tokens, target_tokens):
        if draft_token != target_token:
            break
        accepted += 1
    return accepted


def verify_prefix(draft_tokens: list[int], target_tokens: list[int], scheduled_length: int) -> int:
    if scheduled_length < 0:
        raise ValueError("scheduled_length 不能小于 0")
    return acceptance_length(draft_tokens[:scheduled_length], target_tokens[:scheduled_length])

