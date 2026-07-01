from dspark_nano import (
    DSparkNano,
    DSparkNanoConfig,
    LoadProfile,
    SamplingConfig,
    ScheduledPrefix,
    ToyProfileDrafter,
    acceptance_length,
    schedule_prefix,
)
from dspark_nano.benchmark import compare_drafters


def test_acceptance_length_stops_at_first_target_mismatch():
    draft_tokens = [11, 22, 33, 44]
    target_tokens = [11, 22, 99, 44]

    assert acceptance_length(draft_tokens, target_tokens) == 2


def test_acceptance_length_accepts_full_prefix_when_tokens_match():
    draft_tokens = [1, 2, 3]
    target_tokens = [1, 2, 3]

    assert acceptance_length(draft_tokens, target_tokens) == 3


def test_scheduler_keeps_monotonic_prefix_and_prunes_low_confidence_tail():
    scheduled = schedule_prefix(
        confidences=[0.92, 0.82, 0.64, 0.31],
        load=LoadProfile(name="high", max_verify_tokens=3, min_survival_probability=0.45),
    )

    assert scheduled.length == 3
    assert scheduled.survival_probabilities == [0.92, 0.7544, 0.4828]
    assert scheduled.stop_reason == "max_verify_tokens"
    assert scheduled.rows
    assert scheduled.rows[-1].status == "pruned_by_load"


def test_toy_profile_drafter_has_less_suffix_decay_in_dspark_mode():
    parallel = ToyProfileDrafter(mode="parallel", gamma=5)
    dspark = ToyProfileDrafter(mode="dspark", gamma=5)

    assert dspark.conditional_acceptance()[-1] > parallel.conditional_acceptance()[-1]
    assert dspark.conditional_acceptance()[0] == parallel.conditional_acceptance()[0]


def test_engine_trace_records_draft_confidence_schedule_and_acceptance():
    engine = DSparkNano(DSparkNanoConfig(gamma=5, drafter_mode="dspark"))

    trace = engine.step_trace(prompt_tokens=[1, 2, 3], load="high")

    assert trace.draft_tokens
    assert trace.confidences
    assert 0 < trace.scheduled_length <= 5
    assert 0 <= trace.accepted_length <= trace.scheduled_length
    assert trace.verification_waste == trace.scheduled_length - trace.accepted_length
    assert trace.scheduler_stop_reason
    assert trace.token_rows


def test_generate_returns_text_and_metrics_for_multiple_steps():
    engine = DSparkNano(DSparkNanoConfig(gamma=5, drafter_mode="dspark"))

    result = engine.generate("abc", SamplingConfig(max_tokens=8, load="medium"))

    assert result.text
    assert len(result.token_ids) == 8
    assert result.metrics.rounds >= 1
    assert result.metrics.accepted_tokens == 8
    assert result.metrics.verification_tokens >= result.metrics.accepted_tokens
    assert len(result.traces) == result.metrics.rounds


def test_generate_appends_target_correction_token_after_partial_mismatch():
    engine = DSparkNano(DSparkNanoConfig(gamma=5, drafter_mode="dspark"))

    result = engine.generate("dspark", SamplingConfig(max_tokens=4, load="medium"))
    first_trace = result.traces[0]

    assert first_trace.accepted_length == 3
    assert first_trace.corrected_target_tokens == [first_trace.target_tokens[3]]
    assert first_trace.emitted_tokens == first_trace.draft_tokens[:3] + [first_trace.target_tokens[3]]
    assert result.token_ids[:4] == first_trace.emitted_tokens
    assert result.metrics.corrected_target_tokens == 1


def test_generation_metrics_count_only_tokens_returned_when_final_trace_is_truncated():
    engine = DSparkNano(DSparkNanoConfig(gamma=5, drafter_mode="dspark"))

    result = engine.generate("dspark", SamplingConfig(max_tokens=2, load="medium"))

    assert len(result.token_ids) == 2
    assert result.metrics.accepted_tokens == 2
    assert result.metrics.accepted_draft_tokens == 2
    assert result.metrics.corrected_target_tokens == 0


def test_zero_schedule_fallback_counts_one_target_step():
    engine = DSparkNano(DSparkNanoConfig(gamma=5, drafter_mode="parallel"))

    result = engine.generate(
        [1, 2, 3],
        SamplingConfig(
            max_tokens=1,
            load=LoadProfile(name="zero", max_verify_tokens=0, min_survival_probability=1.0),
        ),
    )
    first_trace = result.traces[0]

    assert first_trace.scheduled_length == 0
    assert first_trace.verification_cost == 1
    assert first_trace.corrected_target_tokens == first_trace.emitted_tokens
    assert result.metrics.verification_tokens == 1


def test_load_profiles_schedule_monotonic_verification_lengths():
    confidences = [0.93, 0.84, 0.76, 0.68, 0.61]

    low = schedule_prefix(confidences, "low")
    medium = schedule_prefix(confidences, "medium")
    high = schedule_prefix(confidences, "high")

    assert low.length >= medium.length >= high.length


def test_benchmark_compares_parallel_and_dspark_modes():
    report = compare_drafters(prompt="dspark", max_tokens=12, gamma=5, load="medium")

    assert set(report) == {"parallel", "dspark"}
    assert report["dspark"].rounds < report["parallel"].rounds
    assert report["dspark"].accepted_tokens == report["parallel"].accepted_tokens == 12
    assert report["parallel"].verification_waste > 0
