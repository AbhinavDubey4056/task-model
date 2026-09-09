
import sys
from predict import predict_next_shipment


def ask_float(prompt, lo=None, hi=None):
    while True:
        raw = input(prompt).strip()
        try:
            val = float(raw)
        except ValueError:
            print("  Please enter a number.")
            continue
        if lo is not None and val < lo:
            print(f"  Must be at least {lo}.")
            continue
        if hi is not None and val > hi:
            print(f"  Must be at most {hi}.")
            continue
        return val


def ask_yes_no(prompt):
    while True:
        raw = input(prompt + " (y/n): ").strip().lower()
        if raw in ("y", "yes"):
            return 1
        if raw in ("n", "no"):
            return 0
        print("  Please answer y or n.")


def ask_priority():
    print("  Priority — 0=low, 1=medium, 2=high")
    return int(ask_float("  Priority (0-2): ", lo=0, hi=2))


def main():
    print("=" * 60)
    print("Shipment delay predictor")
    print("Answer a few questions about the planned feature.")
    print("=" * 60)

    planned_date = input("\nPlanned shipment date (YYYY-MM-DD): ").strip()

    team_size = ask_float("Team size (people): ", lo=1)
    feature_complexity = ask_float("Feature complexity (1-10 scale): ", lo=1, hi=10)
    num_dependencies = ask_float("Number of dependencies on other teams/features: ", lo=0)
    sprint_length_weeks = ask_float("Sprint length (weeks): ", lo=1)
    num_blockers = ask_float("Current number of blockers: ", lo=0)
    holidays_in_sprint = ask_yes_no("Any holidays fall inside this sprint?")
    priority_encoded = ask_priority()
    past_avg_delay_days = ask_float("This team's average past delay (days, can be negative if usually early): ")
    estimated_bug_count = ask_float("Estimated bug count: ", lo=0)

    raw_features = {
        "team_size": team_size,
        "feature_complexity": feature_complexity,
        "num_dependencies": num_dependencies,
        "sprint_length_weeks": sprint_length_weeks,
        "num_blockers": num_blockers,
        "holidays_in_sprint": holidays_in_sprint,
        "priority_encoded": priority_encoded,
        "past_avg_delay_days": past_avg_delay_days,
        "estimated_bug_count": estimated_bug_count,
    }

    try:
        result = predict_next_shipment(planned_date, raw_features)
    except Exception as e:
        print(f"\nCouldn't generate a prediction: {e}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print(f"Planned date:            {result['planned_date']}")
    print(f"Predicted delay:         {result['predicted_delay_days']} days")
    print(f"Predicted shipment date: {result['predicted_shipment_date']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
