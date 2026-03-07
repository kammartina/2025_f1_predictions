"""
F1 Race Prediction using Gradient Boosting Regressor

It learns from 2024 Australian GP race data, then predicts 2025 Australian GP results
based on the qualifying times you provided.

The script is predicting the Australian GP 2025 winner, NOT the Chinese GP winner. Here's the logic:
- Training data: Uses 2024 Australian GP race lap times (historical data from FastF1)
- Input features: Uses 2025 Australian GP qualifying times that you manually entered
- Prediction: Predicts race performance for the Australian GP 2025
"""

import fastf1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error


def load_historical_race_data(year, race_number, session_type="R"):
    """
    Load historical race data from FastF1 API.

    Args:
        year (int): Year of the race (e.g., 2024)
        race_number (int): Race number in the season (e.g., 3 for Australian GP)
        session_type (str): Session type - "R" for Race, "Q" for Qualifying

    Returns:
        pd.DataFrame: DataFrame with Driver and LapTime columns in seconds
    """
    print(f"\n📥 Loading {year} race {race_number} data...")
    session = fastf1.get_session(year, race_number, session_type)
    session.load()

    # Extract lap times
    laps = session.laps[["Driver", "LapTime"]].copy()
    laps.dropna(subset=["LapTime"], inplace=True)
    laps["LapTime (s)"] = laps["LapTime"].dt.total_seconds()

    # Print summary
    print(f"\n📊 Historical Race Data (First 20 laps):\n")
    print(laps.head(20))
    print(f"\nTotal laps fetched: {len(laps)}")
    print(f"Unique drivers: {laps['Driver'].nunique()}")

    return laps


def get_qualifying_data():
    """
    Get 2025 Australian GP qualifying data for all 20 drivers.

    Returns:
        pd.DataFrame: DataFrame with Driver and QualifyingTime columns
    """
    qualifying_data = pd.DataFrame({
        "Driver": [
            "Lando Norris", "Oscar Piastri", "Max Verstappen", "George Russell", "Yuki Tsunoda",
            "Alexander Albon", "Charles Leclerc", "Lewis Hamilton", "Pierre Gasly", "Carlos Sainz",
            "Isack Hadjar", "Fernando Alonso", "Lance Stroll", "Jack Doohan", "Gabriel Bortoleto",
            "Kimi Antonelli", "Nico Hulkenberg", "Liam Lawson", "Esteban Ocon", "Oliver Bearman"
        ],
        "QualifyingTime (s)": [
            75.096, 75.180, 75.481, 75.546, 75.670,
            75.737, 75.755, 75.973, 75.980, 76.062,
            76.175, 76.453, 76.483, 76.863, 77.520,
            76.525, 76.579, 77.094, 77.147, 77.500  # Bearman DNS - using estimated time
        ]
    })

    return qualifying_data


def get_driver_mapping():
    """
    Get mapping from full driver names to FastF1 3-letter codes.

    Returns:
        dict: Dictionary mapping full names to driver codes
    """
    return {
        "Lando Norris": "NOR", "Oscar Piastri": "PIA", "Max Verstappen": "VER", "George Russell": "RUS",
        "Yuki Tsunoda": "TSU", "Alexander Albon": "ALB", "Charles Leclerc": "LEC", "Lewis Hamilton": "HAM",
        "Pierre Gasly": "GAS", "Carlos Sainz": "SAI", "Lance Stroll": "STR", "Fernando Alonso": "ALO",
        "Isack Hadjar": "HAD", "Jack Doohan": "DOO", "Gabriel Bortoleto": "BOR", "Kimi Antonelli": "ANT",
        "Nico Hulkenberg": "HUL", "Liam Lawson": "LAW", "Esteban Ocon": "OCO", "Oliver Bearman": "BEA"
    }


def prepare_training_data(qualifying_df, historical_laps_df, driver_mapping):
    """
    Merge qualifying data with historical race data to create training dataset.

    Args:
        qualifying_df (pd.DataFrame): Qualifying times for 2025
        historical_laps_df (pd.DataFrame): Historical lap times from 2024
        driver_mapping (dict): Mapping from full names to driver codes

    Returns:
        tuple: (X, y) where X is features and y is target lap times
    """
    # Map driver names to codes
    qualifying_df["DriverCode"] = qualifying_df["Driver"].map(driver_mapping)

    # Merge qualifying data with historical race data
    merged_data = qualifying_df.merge(historical_laps_df, left_on="DriverCode", right_on="Driver")

    # Prepare features and target
    X = merged_data[["QualifyingTime (s)"]]
    y = merged_data["LapTime (s)"]

    if X.shape[0] == 0:
        raise ValueError("Dataset is empty after preprocessing. Check data sources!")

    return X, y


def train_model(X, y, test_size=0.2, random_state=39):
    """
    Train a Gradient Boosting Regressor model.

    Args:
        X (pd.DataFrame): Features (qualifying times)
        y (pd.Series): Target (lap times)
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility

    Returns:
        tuple: (trained_model, X_test, y_test) for evaluation
    """
    print("\n🔧 Training Gradient Boosting Model...")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Create and train model
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=random_state)
    model.fit(X_train, y_train)

    print("✅ Model training complete!")

    return model, X_test, y_test


def make_predictions(model, qualifying_df):
    """
    Make race predictions based on qualifying times.

    Args:
        model: Trained Gradient Boosting model
        qualifying_df (pd.DataFrame): Qualifying data with times

    Returns:
        pd.DataFrame: DataFrame with predictions sorted by predicted race time
    """
    print("\n🔮 Making predictions...")

    # Predict lap times
    predicted_lap_times = model.predict(qualifying_df[["QualifyingTime (s)"]])
    qualifying_df["PredictedRaceTime (s)"] = predicted_lap_times

    # Rank drivers by predicted race time
    results = qualifying_df.sort_values(by="PredictedRaceTime (s)").reset_index(drop=True)
    results.insert(0, "Position", range(1, len(results) + 1))

    return results


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance on test data.

    Args:
        model: Trained model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test target values

    Returns:
        float: Mean Absolute Error in seconds
    """
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    return mae


def print_predictions(results_df):
    """
    Print race predictions in a formatted table.

    Args:
        results_df (pd.DataFrame): Results with Position, Driver, and PredictedRaceTime columns
    """
    print("\n🏁 Predicted 2025 Australian GP Winner 🏁\n")
    print(results_df[["Position", "Driver", "PredictedRaceTime (s)"]].to_string(index=False))


def main():
    """Main function to run the F1 prediction pipeline."""
    # Enable caching
    fastf1.Cache.enable_cache("f1_cache")

    # Load historical data
    historical_laps = load_historical_race_data(year=2024, race_number=3, session_type="R")

    # Get qualifying data and driver mapping
    qualifying_2025 = get_qualifying_data()
    driver_mapping = get_driver_mapping()

    # Prepare training data
    X, y = prepare_training_data(qualifying_2025, historical_laps, driver_mapping)

    # Train model
    model, X_test, y_test = train_model(X, y)

    # Make predictions
    results = make_predictions(model, qualifying_2025)

    # Print results
    print_predictions(results)

    # Evaluate model
    mae = evaluate_model(model, X_test, y_test)
    print(f"\n🔍 Model Error (MAE): {mae:.2f} seconds")


if __name__ == "__main__":
    main()
