from __future__ import annotations


class ToyTarget:
    """确定性的 toy target，用来模拟大模型验证分布。

    它不尝试建模自然语言，只提供稳定的“目标 token 序列”，让 DSpark Nano
    可以专注展示 speculative decoding 的控制流。
    """

    def __init__(self, vocab_size: int = 97) -> None:
        if vocab_size < 16:
            raise ValueError("vocab_size 至少为 16")
        self.vocab_size = vocab_size

    def encode(self, text: str) -> list[int]:
        return [(ord(ch) % self.vocab_size) for ch in text]

    def decode(self, token_ids: list[int]) -> str:
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        return "".join(alphabet[token % len(alphabet)] for token in token_ids)

    def next_tokens(self, prompt_tokens: list[int], count: int) -> list[int]:
        if count <= 0:
            return []

        seed = sum(prompt_tokens) + len(prompt_tokens) * 17
        current = prompt_tokens[-1] if prompt_tokens else seed % self.vocab_size
        tokens: list[int] = []
        for index in range(count):
            current = (current * 31 + seed + index * 7 + 11) % self.vocab_size
            tokens.append(current)
        return tokens

