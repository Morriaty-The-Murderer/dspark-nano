from __future__ import annotations

from dspark_nano.config import SamplingConfig
from dspark_nano.engine import DSparkNano, DSparkNanoConfig, GenerationMetrics


def compare_drafters(
    prompt: str = "dspark",
    max_tokens: int = 32,
    gamma: int = 5,
    load: str = "medium",
) -> dict[str, GenerationMetrics]:
    report: dict[str, GenerationMetrics] = {}
    for mode in ("parallel", "dspark"):
        engine = DSparkNano(DSparkNanoConfig(gamma=gamma, drafter_mode=mode))
        result = engine.generate(prompt, SamplingConfig(max_tokens=max_tokens, load=load))
        report[mode] = result.metrics
    return report

