from dspark_nano import compare_drafters


def main() -> None:
    report = compare_drafters(prompt="dspark", max_tokens=24, gamma=5, load="medium")

    print("mode     rounds  tokens  draft  corrected  verified  waste")
    print("-------- ------- ------- ------ ---------- --------- ------")
    for mode, metrics in report.items():
        print(
            f"{mode:8s} "
            f"{metrics.rounds:7d} "
            f"{metrics.accepted_tokens:7d} "
            f"{metrics.accepted_draft_tokens:6d} "
            f"{metrics.corrected_target_tokens:10d} "
            f"{metrics.verification_tokens:9d} "
            f"{metrics.verification_waste:6d}"
        )


if __name__ == "__main__":
    main()
