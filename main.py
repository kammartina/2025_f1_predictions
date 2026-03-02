"""
F1 Prediction Pipeline — CLI entry point.

Commands
────────
  collect   Fetch race data from FastF1 and store in the database
  train     Build feature matrix and train the XGBoost model
  predict   Predict finishing order for an upcoming race
  evaluate  Run leave-one-race-out cross-validation and print accuracy
  summary   Print a summary of what is currently in the database
  features  Show feature importances from the trained model

Examples
────────
  # Collect all 2025 races (run once to populate historical data)
  python main.py collect --year 2025

  # Collect a specific round
  python main.py collect --year 2026 --round 1

  # Collect with telemetry (slow first run, then cached)
  python main.py collect --year 2025 --telemetry

  # Train the model on all data
  python main.py train

  # Train on 2025 + 2026 only
  python main.py train --years 2025 2026

  # Predict an upcoming race (after qualifying is stored)
  python main.py predict --year 2026 --round 1
  python main.py predict --year 2026 --round 1 --air-temp 28 --rainfall 0

  # Evaluate model accuracy
  python main.py evaluate

  # Database summary
  python main.py summary

  # Feature importances
  python main.py features
"""

from __future__ import annotations

import argparse
import logging
import sys

from src.pipeline import F1Pipeline

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sub-command handlers
# ---------------------------------------------------------------------------

def cmd_collect(args: argparse.Namespace, pipeline: F1Pipeline) -> None:
    if args.round:
        pipeline.collect_round(
            year=args.year,
            round_num=args.round,
            include_telemetry=args.telemetry,
            force=args.force,
        )
    else:
        pipeline.collect_season(
            year=args.year,
            include_telemetry=args.telemetry,
            force=args.force,
        )


def cmd_train(args: argparse.Namespace, pipeline: F1Pipeline) -> None:
    years = args.years if args.years else None
    cv = pipeline.train(years=years)
    if not cv.empty:
        print("\nLeave-One-Race-Out CV results:")
        print(cv.to_string(index=False))
        print(
            f"\nMean Spearman r : {cv['spearman_r'].mean():.3f}"
            f"\nMean top-3 hits : {cv['top3_overlap'].mean():.2f} / 3"
        )


def cmd_predict(args: argparse.Namespace, pipeline: F1Pipeline) -> None:
    # Build weather forecast dict from CLI flags (only keys the user provided)
    weather: dict = {}
    weather_map = {
        "air_temp":      args.air_temp,
        "track_temp":    args.track_temp,
        "humidity":      args.humidity,
        "pressure":      args.pressure,
        "rainfall":      args.rainfall,
        "wind_speed":    args.wind_speed,
        "wind_direction": args.wind_direction,
    }
    for key, val in weather_map.items():
        if val is not None:
            weather[key] = val

    predictions = pipeline.predict_race(
        year=args.year,
        round_num=args.round,
        weather_forecast=weather if weather else None,
        auto_weather=not args.no_auto_weather,
    )

    if predictions.empty:
        print("No predictions generated. Check that qualifying data is collected.")
        return

    print(f"\nPredicted finishing order — {args.year} Round {args.round}:")
    print(f"{'Pos':>4}  {'Driver':>6}")
    print("─" * 14)
    for _, row in predictions.iterrows():
        print(f"{int(row['predicted_position']):>4}  {row['driver_code']:>6}")


def cmd_evaluate(args: argparse.Namespace, pipeline: F1Pipeline) -> None:
    years = args.years if args.years else None
    cv = pipeline.evaluate(years=years)
    if cv.empty:
        print("No evaluation data available.")
        return
    print("\nLeave-One-Race-Out CV:")
    print(cv.to_string(index=False))
    print(
        f"\nMean Spearman r : {cv['spearman_r'].mean():.3f}"
        f"\nMean top-3 hits : {cv['top3_overlap'].mean():.2f} / 3"
    )


def cmd_summary(_args: argparse.Namespace, pipeline: F1Pipeline) -> None:
    pipeline.database_summary()


def cmd_features(_args: argparse.Namespace, pipeline: F1Pipeline) -> None:
    try:
        importance = pipeline.feature_importance()
        print("\nFeature importances (XGBoost gain):")
        print(importance.to_string(index=False))
    except RuntimeError as exc:
        print(f"Error: {exc}. Run 'python main.py train' first.")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="f1predict",
        description="F1 race winner prediction pipeline",
    )
    parser.add_argument(
        "--db", default="data/f1_data.db", help="Path to DuckDB database file"
    )
    parser.add_argument(
        "--cache", default="f1_cache", help="FastF1 cache directory"
    )
    parser.add_argument(
        "--model", default="models/predictor.json", help="Model save/load path"
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # collect
    p_collect = sub.add_parser("collect", help="Fetch race data from FastF1")
    p_collect.add_argument("--year", type=int, required=True, help="Season year")
    p_collect.add_argument("--round", type=int, help="Round number (omit = entire season)")
    p_collect.add_argument("--telemetry", action="store_true", help="Also collect telemetry (slow)")
    p_collect.add_argument("--force", action="store_true", help="Re-collect even if data exists")

    # train
    p_train = sub.add_parser("train", help="Train the prediction model")
    p_train.add_argument("--years", type=int, nargs="+", help="Restrict to these seasons")

    # predict
    p_predict = sub.add_parser("predict", help="Predict a race result")
    p_predict.add_argument("--year",  type=int, required=True)
    p_predict.add_argument("--round", type=int, required=True)
    p_predict.add_argument("--air-temp",      type=float, dest="air_temp")
    p_predict.add_argument("--track-temp",    type=float, dest="track_temp")
    p_predict.add_argument("--humidity",      type=float)
    p_predict.add_argument("--pressure",      type=float)
    p_predict.add_argument("--rainfall",      type=int, choices=[0, 1])
    p_predict.add_argument("--wind-speed",    type=float, dest="wind_speed")
    p_predict.add_argument("--wind-direction",  type=float, dest="wind_direction")
    p_predict.add_argument(
        "--no-auto-weather",
        action="store_true",
        dest="no_auto_weather",
        help="Disable automatic Open-Meteo forecast fetching (use historical median instead)",
    )

    # evaluate
    p_eval = sub.add_parser("evaluate", help="Run leave-one-race-out CV")
    p_eval.add_argument("--years", type=int, nargs="+")

    # summary
    sub.add_parser("summary", help="Show database contents summary")

    # features
    sub.add_parser("features", help="Show feature importances")

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    pipeline = F1Pipeline(
        db_path=args.db,
        cache_path=args.cache,
        model_path=args.model,
    )

    handlers = {
        "collect":  cmd_collect,
        "train":    cmd_train,
        "predict":  cmd_predict,
        "evaluate": cmd_evaluate,
        "summary":  cmd_summary,
        "features": cmd_features,
    }

    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    handler(args, pipeline)


if __name__ == "__main__":
    main()
