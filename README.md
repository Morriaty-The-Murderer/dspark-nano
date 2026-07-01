# DSpark Nano

A minimal, readable DSpark-style speculative decoding demo for understanding semi-autoregressive drafting and confidence-scheduled verification.

DSpark Nano 是一个极小、可读、可跑的教学实现，用来解释 DeepSeek DSpark 论文里的核心机制。它是非官方项目，不复用 [DeepSpec](https://github.com/deepseek-ai/DeepSpec) 官方代码，不复现 DeepSeek-V4 serving，也不宣称达到论文中的线上吞吐或速度结论。

## What You Will See

- speculative decoding 的最长前缀接受规则；
- parallel drafter 的 suffix decay；
- DSpark-style profile 如何缓解后缀衰减；
- confidence-scheduled verification 如何减少低价值验证；
- verification waste 如何随 drafter 与 load profile 改变。

Toy benchmark 当前会输出类似结果：

```text
mode     rounds  tokens  draft  corrected  verified  waste
-------- ------- ------- ------ ---------- --------- ------
parallel       8      24     16          8        24      8
dspark         6      24     18          6        24      6
```

这只说明 toy profile 下的机制差异，不是 DeepSpec 或 DeepSeek-V4 的实测结果。

## Quick Start

从仓库根目录运行：

```bash
/opt/homebrew/bin/uv run --cache-dir .uv-cache python -m dspark_nano.cli trace
/opt/homebrew/bin/uv run --cache-dir .uv-cache python -m dspark_nano.cli benchmark
```

也可以运行示例模块：

```bash
/opt/homebrew/bin/uv run --cache-dir .uv-cache python -m examples.toy_trace
/opt/homebrew/bin/uv run --cache-dir .uv-cache python -m examples.benchmark_toy
```

## Python API

```python
from dspark_nano import DSparkNano, DSparkNanoConfig, SamplingConfig

engine = DSparkNano(DSparkNanoConfig(gamma=5, drafter_mode="dspark"))
result = engine.generate("dspark", SamplingConfig(max_tokens=12, load="medium"))

print(result.text)
print(result.metrics)
print(result.traces[0].table())
```

## Code Map

| 文件 | 职责 |
| --- | --- |
| `target.py` | deterministic toy target token 序列 |
| `drafter.py` | oracle toy profile drafter；展示 parallel / dspark suffix decay 差异 |
| `scheduler.py` | DSpark Nano 的 load-aware toy scheduler；不等价于官方 DeepSpec eval 或 DeepSeek-V4 生产 scheduler |
| `verifier.py` | speculative acceptance 最长前缀规则 |
| `trace.py` | 单轮 trace、token 表格和调试数据 |
| `engine.py` | `generate` 与 `step_trace` 主循环 |
| `benchmark.py` | parallel / dspark toy benchmark |
| `cli.py` | `trace` 与 `benchmark` 命令行入口 |

## Trace Preview

```text
pos | draft | target | conf | survival | status
----|-------|--------|------|----------|--------
  0 |     1 |      1 | 0.93 |   0.9300 | accepted
  1 |    20 |     20 | 0.84 |   0.7812 | accepted
  2 |    34 |     34 | 0.76 |   0.5937 | accepted
  3 |    88 |     87 | 0.68 |   0.4037 | corrected
  4 |    89 |     88 | 0.61 |   0.2463 | pruned_by_confidence
```

## Tests

```bash
/opt/homebrew/bin/uv run --cache-dir .uv-cache pytest tests/test_dspark_nano.py
```

当前测试覆盖：

- partial mismatch 会追加 target correction token；
- `scheduled_length == 0` fallback 会计入一次 target step cost；
- load profile 的验证长度单调；
- scheduler stop reason 和 per-token rows；
- trace / generate / benchmark 的核心行为。

## Boundary

`ToyProfileDrafter` 会读取 toy target token，再按手写 acceptance profile 制造命中或 mismatch。这样做是为了让 suffix decay 和 correction 过程可解释、可截图；它不是真实 draft model。后续如果要更接近 DeepSpec，需要另做无 target 泄漏的 toy logits / transition model。
