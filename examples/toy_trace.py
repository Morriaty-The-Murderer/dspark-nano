from dspark_nano import DSparkNano, DSparkNanoConfig, SamplingConfig


def main() -> None:
    engine = DSparkNano(DSparkNanoConfig(gamma=5, drafter_mode="dspark"))
    result = engine.generate("dspark", SamplingConfig(max_tokens=12, load="medium"))
    first_trace = result.traces[0]

    print("DSpark Nano trace")
    print("Generated:", result.text)
    print("Metrics:", result.metrics)
    print()
    print(first_trace.table())
    print()
    print("Emitted:", first_trace.emitted_tokens)
    print("Stop reason:", first_trace.scheduler_stop_reason)


if __name__ == "__main__":
    main()
