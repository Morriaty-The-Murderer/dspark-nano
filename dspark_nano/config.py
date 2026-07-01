from dataclasses import dataclass

from dspark_nano.scheduler import LoadProfile


@dataclass(frozen=True)
class SamplingConfig:
    """生成配置，保持接近常见 LLM.generate API。"""

    max_tokens: int = 32
    load: str | LoadProfile = "medium"

    def __post_init__(self) -> None:
        if self.max_tokens <= 0:
            raise ValueError("max_tokens 必须大于 0")
