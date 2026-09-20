import os
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = "TrafficDataset.csv"

# For Kaggle traffic dataset
TIMESTAMP_COLUMN = "DateTime"
TRAFFIC_COLUMN = "Vehicles"
JUNCTION_COLUMN = "Junction"
SELECTED_JUNCTION = 1

# Report requirement
T_IN = 12        # previous 12 time steps
T_OUT = 5       # predict next 5 time steps H0-H4

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.2

EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.001
PATIENCE = 8

RANDOM_SEED = 42

OUTPUT_DIR = "outputs"


# ============================================================
# SETUP
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(RANDOM_SEED)


# ============================================================
# DATA LOADING
# ============================================================

def load_dataset():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"{DATASET_PATH} not found. Put TrafficDataset.csv in the same folder as main.py"
        )

    df = pd.read_csv(DATASET_PATH)

    print("\nDataset loaded successfully")
    print("Dataset shape:", df.shape)
    print("Dataset columns:", list(df.columns))

    required_columns = [TIMESTAMP_COLUMN, TRAFFIC_COLUMN, JUNCTION_COLUMN]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in dataset.")

    # Convert DateTime column
    df[TIMESTAMP_COLUMN] = pd.to_datetime(df[TIMESTAMP_COLUMN], errors="coerce")

    # Remove invalid dates
    df = df.dropna(subset=[TIMESTAMP_COLUMN])

    # Select one junction for clean univariate time-series forecasting
    df = df[df[JUNCTION_COLUMN] == SELECTED_JUNCTION]

    # Sort chronologically
    df = df.sort_values(by=TIMESTAMP_COLUMN)

    # Select traffic column
    traffic = pd.to_numeric(df[TRAFFIC_COLUMN], errors="coerce")

    # Handle missing values
    traffic = traffic.ffill()
    traffic = traffic.fillna(0)

    values = traffic.values.astype(np.float32)

    print(f"\nUsing Junction: {SELECTED_JUNCTION}")
    print("Total records after filtering:", len(values))
    print("First 10 vehicle values:", values[:10])

    if len(values) < 200:
        raise ValueError("Dataset too small after filtering. Choose another junction or use full dataset.")

    return values, df


# ============================================================
# SEQUENCE GENERATION
# ============================================================

def create_sequences(values, t_in, t_out):
    X = []
    Y = []

    for i in range(len(values) - t_in - t_out + 1):
        X.append(values[i:i + t_in])
        Y.append(values[i + t_in:i + t_in + t_out])

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)

    return X, Y


# ============================================================
# TRAIN VALIDATION TEST SPLIT
# ============================================================

def split_data(X, Y):
    total_samples = len(X)

    train_end = int(total_samples * TRAIN_RATIO)
    val_end = int(total_samples * (TRAIN_RATIO + VAL_RATIO))

    X_train = X[:train_end]
    Y_train = Y[:train_end]

    X_val = X[train_end:val_end]
    Y_val = Y[train_end:val_end]

    X_test = X[val_end:]
    Y_test = Y[val_end:]

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_data(X_train, Y_train, X_val, Y_val, X_test, Y_test):
    mean = X_train.mean()
    std = X_train.std()

    if std == 0:
        std = 1.0

    X_train = (X_train - mean) / std
    Y_train = (Y_train - mean) / std

    X_val = (X_val - mean) / std
    Y_val = (Y_val - mean) / std

    X_test = (X_test - mean) / std
    Y_test = (Y_test - mean) / std

    return X_train, Y_train, X_val, Y_val, X_test, Y_test, mean, std


def denormalize(data, mean, std):
    return data * std + mean


# ============================================================
# DATALOADERS
# ============================================================

def create_dataloaders(X_train, Y_train, X_val, Y_val, X_test, Y_test):
    X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
    Y_train = torch.tensor(Y_train, dtype=torch.float32)

    X_val = torch.tensor(X_val, dtype=torch.float32).unsqueeze(-1)
    Y_val = torch.tensor(Y_val, dtype=torch.float32)

    X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
    Y_test = torch.tensor(Y_test, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_train, Y_train),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        TensorDataset(X_val, Y_val),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_loader = DataLoader(
        TensorDataset(X_test, Y_test),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return train_loader, val_loader, test_loader, X_test, Y_test


# ============================================================
# MODELS
# ============================================================

class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
            batch_first=True
        )

        self.fc = nn.Linear(HIDDEN_SIZE, T_OUT)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


class GRUModel(nn.Module):
    def __init__(self):
        super(GRUModel, self).__init__()

        self.gru = nn.GRU(
            input_size=1,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
            batch_first=True
        )

        self.fc = nn.Linear(HIDDEN_SIZE, T_OUT)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


class HybridLSTMGRUModel(nn.Module):
    def __init__(self):
        super(HybridLSTMGRUModel, self).__init__()

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
            batch_first=True
        )

        self.gru = nn.GRU(
            input_size=1,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
            batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Linear(HIDDEN_SIZE * 2, 128),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(128, T_OUT)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        gru_out, _ = self.gru(x)

        lstm_last = lstm_out[:, -1, :]
        gru_last = gru_out[:, -1, :]

        combined = torch.cat((lstm_last, gru_last), dim=1)

        out = self.fc(combined)
        return out


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_model(model, model_name, train_loader, val_loader):
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")
    patience_counter = 0

    train_losses = []
    val_losses = []

    best_model_path = os.path.join(OUTPUT_DIR, f"{model_name}_best.pth")

    print(f"\nTraining {model_name} Model")

    for epoch in range(EPOCHS):
        model.train()

        total_train_loss = 0.0

        for batch_X, batch_Y in train_loader:
            batch_X = batch_X.to(device)
            batch_Y = batch_Y.to(device)

            optimizer.zero_grad()

            predictions = model(batch_X)
            loss = criterion(predictions, batch_Y)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()

        total_val_loss = 0.0

        with torch.no_grad():
            for batch_X, batch_Y in val_loader:
                batch_X = batch_X.to(device)
                batch_Y = batch_Y.to(device)

                predictions = model(batch_X)
                loss = criterion(predictions, batch_Y)

                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Train Loss: {avg_train_loss:.6f} "
            f"Validation Loss: {avg_val_loss:.6f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"Early stopping applied for {model_name}")
            break

    model.load_state_dict(torch.load(best_model_path, map_location=device))

    return model, train_losses, val_losses


# ============================================================
# PREDICTION
# ============================================================

def predict(model, X_test):
    model.eval()

    predictions = []

    with torch.no_grad():
        X_test = X_test.to(device)

        for i in range(0, len(X_test), 256):
            batch_X = X_test[i:i + 256]
            output = model(batch_X)
            predictions.append(output.cpu().numpy())

    predictions = np.vstack(predictions)

    return predictions


# ============================================================
# METRICS
# ============================================================

def evaluate_model(y_true, y_pred, model_name):
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)

    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    rmse = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
    r2 = r2_score(y_true_flat, y_pred_flat)

    print(f"\n{model_name} Overall Results")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R2   : {r2:.4f}")

    horizon_results = []

    for h in range(T_OUT):
        h_mae = mean_absolute_error(y_true[:, h], y_pred[:, h])
        h_rmse = np.sqrt(mean_squared_error(y_true[:, h], y_pred[:, h]))
        h_r2 = r2_score(y_true[:, h], y_pred[:, h])

        horizon_results.append([model_name, f"H{h}", h_mae, h_rmse, h_r2])

        print(
            f"{model_name} H{h} -> "
            f"MAE: {h_mae:.4f}, "
            f"RMSE: {h_rmse:.4f}, "
            f"R2: {h_r2:.4f}"
        )

    return {
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }, horizon_results


# ============================================================
# PLOTS
# ============================================================

def plot_loss(train_losses, val_losses, model_name):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title(f"{model_name} Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{model_name}_loss_curve.png"), dpi=300)
    plt.close()


def plot_predictions(y_true, predictions_dict):
    n = min(500, len(y_true))

    plt.figure(figsize=(14, 6))
    plt.plot(y_true[:n, 0], label="Actual H0")

    for model_name, y_pred in predictions_dict.items():
        plt.plot(y_pred[:n, 0], label=f"{model_name} Predicted H0")

    plt.title("Actual vs Predicted Traffic Flow")
    plt.xlabel("Test Sample")
    plt.ylabel("Vehicles")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "actual_vs_predicted_H0.png"), dpi=300)
    plt.close()


def plot_scatter(y_true, predictions_dict):
    for model_name, y_pred in predictions_dict.items():
        plt.figure(figsize=(6, 6))
        plt.scatter(y_true[:, 0], y_pred[:, 0], alpha=0.5)

        min_val = min(y_true[:, 0].min(), y_pred[:, 0].min())
        max_val = max(y_true[:, 0].max(), y_pred[:, 0].max())

        plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

        plt.title(f"{model_name} Scatter Plot H0")
        plt.xlabel("Actual Vehicles")
        plt.ylabel("Predicted Vehicles")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{model_name}_scatter_H0.png"), dpi=300)
        plt.close()


def plot_residual_histogram(y_true, predictions_dict):
    for model_name, y_pred in predictions_dict.items():
        residual = y_true[:, 0] - y_pred[:, 0]

        plt.figure(figsize=(8, 5))
        plt.hist(residual, bins=40)
        plt.title(f"{model_name} Residual Histogram H0")
        plt.xlabel("Residual Error")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{model_name}_residual_H0.png"), dpi=300)
        plt.close()


def plot_cumulative_error(y_true, predictions_dict):
    plt.figure(figsize=(12, 6))

    for model_name, y_pred in predictions_dict.items():
        error = np.abs(y_true[:, 0] - y_pred[:, 0])
        cumulative_error = np.cumsum(error)
        plt.plot(cumulative_error, label=model_name)

    plt.title("Cumulative Absolute Error H0")
    plt.xlabel("Test Sample")
    plt.ylabel("Cumulative Error")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "cumulative_error_H0.png"), dpi=300)
    plt.close()


def plot_horizon_mae(horizon_df):
    plt.figure(figsize=(10, 6))

    for model_name in horizon_df["Model"].unique():
        temp = horizon_df[horizon_df["Model"] == model_name]
        plt.plot(temp["Horizon"], temp["MAE"], marker="o", label=model_name)

    plt.title("Horizon-wise MAE Comparison")
    plt.xlabel("Prediction Horizon")
    plt.ylabel("MAE")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "horizon_wise_mae.png"), dpi=300)
    plt.close()


# ============================================================
# CASE STUDY
# ============================================================

def classify_traffic(value, low_threshold, high_threshold):
    if value < low_threshold:
        return "Low Traffic"
    elif value < high_threshold:
        return "Medium Traffic"
    else:
        return "High Traffic"


def case_study(X_test_original, y_true, predictions_dict):
    index = 0

    low_threshold = np.percentile(y_true.reshape(-1), 33)
    high_threshold = np.percentile(y_true.reshape(-1), 66)

    input_values = X_test_original[index]
    actual_values = y_true[index]

    print("\n================ CASE STUDY / TEST CASE ================")
    print("Input: Previous 12 hourly vehicle counts")
    print(np.round(input_values, 2))

    print("\nActual next 5 hourly vehicle counts H0-H4")
    print(np.round(actual_values, 2))

    case_file = os.path.join(OUTPUT_DIR, "case_study_result.txt")

    with open(case_file, "w") as f:
        f.write("CASE STUDY / TEST CASE RESULT\n")
        f.write("=============================\n\n")

        f.write("Input: Previous 12 hourly vehicle counts\n")
        f.write(str(np.round(input_values, 2)))
        f.write("\n\n")

        f.write("Actual next 5 hourly vehicle counts H0-H4\n")
        f.write(str(np.round(actual_values, 2)))
        f.write("\n\n")

        for model_name, y_pred in predictions_dict.items():
            predicted_values = y_pred[index]

            print(f"\n{model_name} predicted next 5 values H0-H4")
            print(np.round(predicted_values, 2))

            f.write(f"{model_name} predicted next 5 values H0-H4\n")
            f.write(str(np.round(predicted_values, 2)))
            f.write("\n")

            print(f"{model_name} congestion classification:")
            f.write(f"{model_name} congestion classification:\n")

            for h in range(T_OUT):
                level = classify_traffic(predicted_values[h], low_threshold, high_threshold)

                print(f"H{h}: {predicted_values[h]:.2f} -> {level}")
                f.write(f"H{h}: {predicted_values[h]:.2f} -> {level}\n")

            f.write("\n")

    print("\nCase study saved to:", case_file)


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():
    print("====================================================")
    print("AI-Powered Traffic Congestion Prediction")
    print("LSTM, GRU and Hybrid LSTM-GRU")
    print("====================================================")
    print("Device:", device)

    # Load data
    values, df = load_dataset()

    # Create sequences
    X, Y = create_sequences(values, T_IN, T_OUT)

    print("\nSequence generation completed")
    print("X shape:", X.shape)
    print("Y shape:", Y.shape)
    print("Input shape per sample:", (T_IN, 1))
    print("Output shape per sample:", (T_OUT,))

    # Split
    X_train, Y_train, X_val, Y_val, X_test, Y_test = split_data(X, Y)

    print("\nData split completed")
    print("Training samples:", len(X_train))
    print("Validation samples:", len(X_val))
    print("Testing samples:", len(X_test))

    # Save original test input for case study
    X_test_original = X_test.copy()

    # Normalize
    X_train, Y_train, X_val, Y_val, X_test, Y_test, mean, std = normalize_data(
        X_train, Y_train, X_val, Y_val, X_test, Y_test
    )

    print("\nNormalization completed")
    print("Mean:", round(float(mean), 4))
    print("Std:", round(float(std), 4))

    # Dataloaders
    train_loader, val_loader, test_loader, X_test_tensor, Y_test_tensor = create_dataloaders(
        X_train, Y_train, X_val, Y_val, X_test, Y_test
    )

    # Define models
    models = {
        "LSTM": LSTMModel(),
        "GRU": GRUModel(),
        "Hybrid_LSTM_GRU": HybridLSTMGRUModel()
    }

    trained_models = {}
    predictions_dict = {}

    summary_results = []
    horizon_results_all = []

    # Train, validate and test each model
    for model_name, model in models.items():
        trained_model, train_losses, val_losses = train_model(
            model,
            model_name,
            train_loader,
            val_loader
        )

        trained_models[model_name] = trained_model

        plot_loss(train_losses, val_losses, model_name)

        pred_norm = predict(trained_model, X_test_tensor)

        pred_original = denormalize(pred_norm, mean, std)
        pred_original = np.clip(pred_original, 0, None)

        predictions_dict[model_name] = pred_original

    # Denormalize true test data
    y_true_original = denormalize(Y_test, mean, std)
    y_true_original = np.clip(y_true_original, 0, None)

    # Evaluation
    print("\n================ FINAL TESTING RESULTS ================")

    for model_name, y_pred in predictions_dict.items():
        summary, horizon_results = evaluate_model(
            y_true_original,
            y_pred,
            model_name
        )

        summary_results.append(summary)
        horizon_results_all.extend(horizon_results)

    summary_df = pd.DataFrame(summary_results)
    horizon_df = pd.DataFrame(
        horizon_results_all,
        columns=["Model", "Horizon", "MAE", "RMSE", "R2"]
    )

    summary_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison_summary.csv"), index=False)
    horizon_df.to_csv(os.path.join(OUTPUT_DIR, "horizon_wise_metrics.csv"), index=False)

    print("\nModel Comparison Summary")
    print(summary_df)

    print("\nHorizon-wise Metrics")
    print(horizon_df)

    # Save predictions
    prediction_df = pd.DataFrame()

    for h in range(T_OUT):
        prediction_df[f"Actual_H{h}"] = y_true_original[:, h]

    for model_name, y_pred in predictions_dict.items():
        for h in range(T_OUT):
            prediction_df[f"{model_name}_Pred_H{h}"] = y_pred[:, h]

    prediction_df.to_csv(os.path.join(OUTPUT_DIR, "test_predictions.csv"), index=False)

    # Plots
    plot_predictions(y_true_original, predictions_dict)
    plot_scatter(y_true_original, predictions_dict)
    plot_residual_histogram(y_true_original, predictions_dict)
    plot_cumulative_error(y_true_original, predictions_dict)
    plot_horizon_mae(horizon_df)

    # Case study
    case_study(X_test_original, y_true_original, predictions_dict)

    print("\n====================================================")
    print("Execution completed successfully")
    print("Check the outputs folder for:")
    print("1. Model results CSV")
    print("2. Horizon-wise metrics")
    print("3. Prediction plots")
    print("4. Case study result")
    print("====================================================")


if __name__ == "__main__":
    main()
