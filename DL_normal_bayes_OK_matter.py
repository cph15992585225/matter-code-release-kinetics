import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import os
import seaborn as sns
from scipy.stats import skew
import json
import optuna
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'

def get_original_scale_data(scaler, data):

    if scaler is None:
        return data
    return scaler.inverse_transform(data.reshape(-1, 1)).flatten()
def inverse_scale(scaler, data):

    if scaler is None:
        return data
    return scaler.inverse_transform(data.reshape(-1, 1)).flatten()


def plot_true_vs_prediction(y_true_orig, y_pred_orig, model_name, current_dir):

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.figure(figsize=(16, 8))

    indices = np.arange(len(y_true_orig))

    plt.plot(indices, y_true_orig, 'o-', color='#1f77b4',
             markersize=8, linewidth=2, label='True Values (Original Scale)')
    plt.plot(indices, y_pred_orig, 's--', color='#d62728',
             markersize=8, linewidth=2, label='Predicted Values (Original Scale)')

    plt.xlabel("Sample Index", fontsize=18, fontweight='bold')
    plt.ylabel("Value", fontsize=18, fontweight='bold')
    plt.title(f"{model_name} - Prediction Results (Original Scale)",
              fontsize=20, fontweight='bold', pad=20)
    plt.legend(fontsize=16, frameon=False)
    plt.grid(True, alpha=0.3)

    plt.xlim(-1, len(y_true_orig) + 1)
    all_values = np.concatenate([y_true_orig, y_pred_orig])
    plt.ylim(all_values.min() * 0.9, all_values.max() * 1.1)

    save_dir = os.path.join(current_dir, "figures", model_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{model_name}_original_scale_comparison.tiff")

    plt.savefig(save_path, format='tiff', dpi=600, bbox_inches='tight')
    plt.close()
    print(f"The original scale comparison chart has been saved as follows:{save_path}")


def read_excel_data(file_path, use_header=True):

    if use_header:
        df = pd.read_excel(file_path, header=0)
    else:
        df = pd.read_excel(file_path, header=None)

    if df.shape[1] < 3:
        raise ValueError("The Excel file must contain at least three columns: the first column is the sequence number, the last column is the output, and the middle column is the feature value.")

    index = df.iloc[:, 0]
    features = df.iloc[:, 1:-1]
    output = df.iloc[:, -1]

    return index, features, output


def prepare_dataloaders(features, labels,
                        test_size=0.1,
                        batch_size=32,
                        random_state=42,
                        shuffle=True,
                        normalize_output=True):

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        features.values,
        labels.values.reshape(-1, 1) if normalize_output else labels.values,
        test_size=test_size,
        random_state=random_state
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.2,
        random_state=random_state
    )

    feature_scaler = StandardScaler().fit(X_train)
    label_scaler = StandardScaler().fit(y_train) if normalize_output else None

    def scale_data(X, y):
        return (
            feature_scaler.transform(X),
            label_scaler.transform(y) if normalize_output else y
        )

    X_train_scaled, y_train_scaled = scale_data(X_train, y_train)
    X_val_scaled, y_val_scaled = scale_data(X_val, y_val)
    X_test_scaled, y_test_scaled = scale_data(X_test, y_test)

    def create_loader(X, y, shuffle):
        dataset = TensorDataset(
            torch.FloatTensor(X),
            torch.FloatTensor(y.squeeze())
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    loaders = (
        create_loader(X_train_scaled, y_train_scaled, shuffle=True),
        create_loader(X_val_scaled, y_val_scaled, shuffle=False),
        create_loader(X_test_scaled, y_test_scaled, shuffle=False),
        feature_scaler,
        label_scaler
    )



    return loaders

def plot_train_vs_test_scatter(y_train_true, y_train_pred,
                              y_test_true, y_test_pred,
                              model_name,
                              train_metrics, test_metrics,
                              current_dir):

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    ax.grid(False)

    all_true = np.concatenate([y_train_true, y_test_true])
    all_pred = np.concatenate([y_train_pred, y_test_pred])
    min_val = min(all_true.min(), all_pred.min())
    max_val = max(all_true.max(), all_pred.max())
    margin = (max_val - min_val) * 0.05
    axis_limits = [min_val - margin, max_val + margin]

    train_scatter = ax.scatter(
        y_train_true, y_train_pred,
        edgecolor='#6A9ACE', facecolor='none',
        marker='o', s=120, linewidths=2,
        label='Train Samples'
    )

    test_scatter = ax.scatter(
        y_test_true, y_test_pred,
        edgecolor='#F1766D', facecolor='none',
        marker='o', s=120, linewidths=2,
        label='Test Samples'
    )

    train_slope, train_intercept = np.polyfit(y_train_true, y_train_pred, 1)
    train_line = train_slope * all_true + train_intercept
    ax.plot(all_true, train_line,
            color='#6A9ACE', lw=2,
            label=f'Train Regression (R²={train_metrics["R²"]:.3f})')

    test_slope, test_intercept = np.polyfit(y_test_true, y_test_pred, 1)
    test_line = test_slope * all_true + test_intercept
    ax.plot(all_true, test_line,
            color='#F1766D', lw=2, linestyle='--',
            label=f'Test Regression (R²={test_metrics["R²"]:.3f})')

    ax.plot(axis_limits, axis_limits, 'k--', lw=2,
            label='Perfect Prediction')

    ax.set_xlim([-5, 100])
    ax.set_ylim(axis_limits)
    ax.set_xlabel('True Value', fontsize=18, fontweight='bold')
    ax.set_ylabel('Predicted Value', fontsize=18, fontweight='bold')
    ax.set_title(f'{model_name}', fontsize=20, fontweight='bold')
    ax.tick_params(axis='both', labelsize=16)

    train_metrics_text = (
        "Train Metrics:\n"
        f"R² = {train_metrics['R²']:.4f}\n"
        f"RMSE = {train_metrics['RMSE']:.4f}\n"
        f"MSE = {train_metrics['MSE']:.4f}\n"
        f"MAE = {train_metrics['MAE']:.4f}"
    )
    plt.text(0.05, 0.95, train_metrics_text, transform=ax.transAxes,
             fontsize=18, fontweight='bold', verticalalignment='top', color='#6A9ACE',
             horizontalalignment='left')

    test_metrics_text = (
        "Test Metrics:\n"
        f"R² = {test_metrics['R²']:.4f}\n"
        f"RMSE = {test_metrics['RMSE']:.4f}\n"
        f"MSE = {test_metrics['MSE']:.4f}\n"
        f"MAE = {test_metrics['MAE']:.4f}"
    )
    plt.text(0.95, 0.05, test_metrics_text, transform=ax.transAxes,
             fontsize=18, fontweight='bold', verticalalignment='bottom', color='#F1766D',
             horizontalalignment='right')

    save_dir = f'{current_dir}/figures/{model_name}'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{model_name}_train_vs_test_scatter.tiff")
    plt.savefig(save_path, format='tiff', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Train vs Test scatter plot saved to: {save_path}")

def plot_regression_scatter(y_true, y_pred, model_name, dataset_type, metrics):

    if dataset_type == 'val':
        dataset_type = 'train'

    plt.rcParams['font.family'] = 'Times New Roman'

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    plt.figure(figsize=(8, 6))
    ax = plt.gca()


    if dataset_type == 'train':
        scatter_color = '#6A9ACE'
        marker_style = 'o'
        face_color = 'none'
    elif dataset_type == 'test':
        scatter_color = '#F1766D'
        marker_style = 'o'
        face_color = 'none'
    else:
        scatter_color = 'gray'
        marker_style = 'o'
        face_color = 'none'

    plt.scatter(y_true, y_pred, alpha=0.8, label='Samples',
                edgecolor=scatter_color, facecolor=face_color,
                marker=marker_style, s=100, linewidths=1.5)

    slope, intercept = np.polyfit(y_true, y_pred, 1)
    reg_line = slope * y_true + intercept
    r2 = r2_score(y_true, y_pred)

    plt.plot(y_true, reg_line, color=scatter_color,
             label=f'Regression Line (R²={r2:.3f})', linewidth=2)
    plt.plot([y_true.min(), y_true.max()],
             [y_true.min(), y_true.max()],
             'k--', lw=2, label='Perfect Prediction')

    ax.set_xlabel('True Value', fontsize=18, fontweight='bold')
    ax.set_ylabel('Predicted Value', fontsize=18, fontweight='bold')
    ax.set_title(model_name, fontsize=20, fontweight='bold')

    ax.grid(False)

    ax.legend().set_visible(False)

    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)

    metrics_text = (
        f"{dataset_type.capitalize()} Metrics:\n"
        f"R² = {metrics['R²']:.4f}\n"
        f"RMSE = {metrics['RMSE']:.4f}\n"
        f"MSE = {metrics['MSE']:.4f}\n"
        f"MAE = {metrics['MAE']:.4f}"
    )
    plt.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
             fontsize=18, fontweight='bold', verticalalignment='top',
             horizontalalignment='left')

    save_dir = os.path.join(current_dir, "figures", model_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{model_name}_{dataset_type}_scatter.tiff")
    plt.savefig(save_path, format='tiff', dpi=300, bbox_inches='tight')

    plt.close()
    print(f"{dataset_type.capitalize()} set scatter plot saved to: {save_path}")

def plot_combined_error_distribution(y_train_true, y_train_pred, y_val_true, y_val_pred, y_test_true, y_test_pred, model_name, current_dir):

    from matplotlib.font_manager import FontProperties

    os.makedirs(os.path.join(current_dir, "figures"), exist_ok=True)

    y_train_true = np.concatenate([y_train_true, y_val_true])
    y_train_pred = np.concatenate([y_train_pred, y_val_pred])

    errors = {
        'Train': np.array(y_train_true) - np.array(y_train_pred),
        'Test': np.array(y_test_true) - np.array(y_test_pred)
    }

    fig, ax1 = plt.subplots(figsize=(12, 8))
    ax2 = ax1.twinx()
    plt.rcParams['font.family'] = 'Times New Roman'

    color_palette = {
        'Train': ('#1f77b4', '#6A9ACE'),
        'Test': ('#d62728', '#F1766D')
    }

    kde_lines = []

    for dataset, err in errors.items():
        kde_color, hist_color = color_palette[dataset]

        sns.histplot(err,
                     ax=ax1,
                     color=hist_color,
                     alpha=0.3,
                     bins=30,
                     kde=False,
                     edgecolor='none',
                     linewidth=0.5,
                     label=f'{dataset} Histogram')

    for dataset, err in errors.items():
        kde_color, hist_color = color_palette[dataset]

        kde_plot = sns.kdeplot(err,
                               ax=ax2,
                               color=kde_color,
                               linewidth=2.5,
                               label=f'{dataset} KDE')
        kde_lines.append(kde_plot.get_lines()[-1])

    ax1.set_xlabel("Prediction Error", fontsize=18, fontweight='bold', labelpad=10)
    ax1.set_ylabel("Frequency", fontsize=18, fontweight='bold', labelpad=10)
    ax2.set_ylabel("Density", fontsize=18, fontweight='bold', labelpad=10, rotation=270, va='bottom')

    all_errors = np.concatenate(list(errors.values()))
    x_min, x_max = all_errors.min(), all_errors.max()
    ax1.set_xlim(x_min * 1.1, x_max * 1.1)
    ax2.spines['right'].set_position(('axes', 1))
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2 = kde_lines
    labels2 = [l.get_label() for l in handles2]
    seen_labels = set()
    unique_handles = []
    unique_labels = []
    for h, l in zip(handles1 + handles2, labels1 + labels2):
        if l not in seen_labels:
            seen_labels.add(l)
            unique_handles.append(h)
            unique_labels.append(l)
    legend_font = FontProperties(
        family='Times New Roman',
        weight='bold',
        size=18
    )
    title_font = FontProperties(
        family='Times New Roman',
        weight='bold',
        size=18
    )

    ax1.legend(unique_handles, unique_labels,
               loc='upper left',
               prop=legend_font,
               title='Dataset & Type',
               title_fontproperties=title_font,
               frameon=False)

    stats_text = ""
    for dataset, err in errors.items():
        stats_text += (
            f"{dataset} set:\n"
            f"μ = {err.mean():.3f}\n"
            f"σ = {err.std():.3f}\n"
            f"Skew = {skew(err):.2f}\n\n"
        )

    ax1.text(0.85, 0.97, stats_text,
             transform=ax1.transAxes,
             ha='left',
             va='top',
             fontsize=18,
             bbox=dict(facecolor='none', edgecolor='white', boxstyle='round'))

    plt.title(f"Dual-Axis Error Analysis - {model_name}",
              fontsize=20, fontweight='bold', pad=20)
    ax1.tick_params(axis='both', labelsize=16)
    ax2.tick_params(axis='y', labelsize=16)

    save_dir = os.path.join(current_dir, "figures", model_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{model_name}_dual_axis_error_dist.tiff")
    plt.savefig(save_path, format='tiff', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Dual-axis error distribution saved: {save_path}")


def plot_training_curves(train_losses, val_losses, r2_scores, rmse_scores, mae_scores):

    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss", marker='o', linestyle='-')
    plt.plot(epochs, val_losses, label="Validation Loss", marker='s', linestyle='--')
    plt.xlabel("Epochs",fontsize=18,fontweight='bold')
    plt.ylabel("Loss",fontsize=18,fontweight='bold')
    plt.title("Loss Curve",fontsize=20,fontweight='bold')
    plt.legend(fontsize=16)
    plt.grid(False)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, r2_scores, label="R² Score", marker='o', linestyle='-')
    plt.plot(epochs, rmse_scores, label="RMSE", marker='s', linestyle='--')
    plt.plot(epochs, mae_scores, label="MAE", marker='^', linestyle=':')
    #plt.plot(epochs, mae_scores, label="MSE", marker='D', linestyle=':')
    plt.xlabel("Epochs", fontsize=18, fontweight='bold')
    plt.ylabel("Metric Value", fontsize=18, fontweight='bold')
    plt.title("Validation Metrics", fontsize=20, fontweight='bold')
    plt.tick_params(labelsize=16)
    plt.legend(fontsize=16)
    plt.grid(False)

    plt.tight_layout()
    save_dir = os.path.join(current_dir, "figures", model_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{model_name}_train_val_fig.tiff")
    plt.savefig(save_path, format='tiff', dpi=300)


def compute_permutation_importance(model, test_loader, device, feature_names, n_repeats=5, random_state=42):

    X_test, y_test = [], []
    for batch in test_loader:
        features, labels = batch
        X_test.append(features.numpy())
        y_test.append(labels.numpy())
    X_test = np.concatenate(X_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)

    from sklearn.base import BaseEstimator
    class PyTorchWrapper(BaseEstimator):
        def __init__(self, model, device):
            super().__init__()
            self.model = model.to(device)
            self.device = device
            self.model.eval()

        def fit(self, X, y):
            return self

        def predict(self, X):
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to(self.device)
                if isinstance(self.model, LSTMModel) and X_tensor.dim() == 2:
                    X_tensor = X_tensor.unsqueeze(1)
                output = self.model(X_tensor)
                return output.cpu().numpy().flatten()

    from sklearn.inspection import permutation_importance
    wrapped_model = PyTorchWrapper(model, device)

    result = permutation_importance(
        estimator=wrapped_model,
        X=X_test,
        y=y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring=lambda est, X, y: -mean_absolute_error(y, est.predict(X))
    )

    sorted_idx = result.importances_mean.argsort()[::-1]
    return (
        np.array(feature_names)[sorted_idx],
        result.importances_mean[sorted_idx]
    )


def plot_permutation_importance(feature_names, scores, model_name, current_dir):

    plt.rcParams['font.family'] = 'Times New Roman'

    TOP_N = 15
    PRIMARY_COLOR = '#084594'
    FIG_SIZE = (10, 8)

    top_features = feature_names[:TOP_N][::-1]
    top_scores = scores[:TOP_N][::-1]

    plt.figure(figsize=FIG_SIZE)
    ax = plt.gca()

    bars = ax.barh(range(TOP_N), top_scores,
                   height=0.75,
                   color=PRIMARY_COLOR,
                   edgecolor='none')

    ax.set_yticks(range(TOP_N))
    ax.set_yticklabels(top_features, fontsize=16, fontweight='bold')
    ax.set_xlabel('Permutation Importance', fontsize=18, fontweight='bold', labelpad=15)
    ax.set_title(f'Top {TOP_N} Features - {model_name}',
                 fontsize=20, fontweight='bold', pad=20)

    ax.spines['right'].set_visible(1.2)
    ax.spines['top'].set_visible(1.2)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.tick_params(axis='both', which='major',
                   width=1.2, length=6,
                   labelsize=16)

    save_path = f'{current_dir}/figures/{model_name}/{model_name}_perm_importance.tiff'
    plt.savefig(save_path, dpi=600, bbox_inches='tight', format='tiff')
    plt.close()


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_layers, dropout_rate=0.3, activation=nn.ReLU()):

        super(MLP, self).__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.Dropout(dropout_rate))
            layers.append(activation)
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x).squeeze()
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.lstm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        prev_dim = input_dim

        for idx, hidden_dim in enumerate(hidden_dims):

            self.lstm_layers.append(
                nn.LSTM(
                    input_size=prev_dim,
                    hidden_size=hidden_dim,
                    num_layers=1,
                    batch_first=True
                )
            )
            if idx < len(hidden_dims) - 1:
                self.dropout_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.fc = nn.Linear(prev_dim, 1)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        for i in range(len(self.lstm_layers)):
            x, _ = self.lstm_layers[i](x)
            if i < len(self.lstm_layers) - 1:
                x = self.dropout_layers[i](x)

        out = self.fc(x[:, -1, :])
        return out.squeeze()
class CNNModel(nn.Module):
    def __init__(self, input_dim, num_filters=64, kernel_size=3, dropout=0.3, padding=1):

        super(CNNModel, self).__init__()

        if kernel_size > input_dim:
            raise ValueError(f"kernel_size ({kernel_size}) 不能超过 input_dim ({input_dim})")

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding=padding
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.output_length = input_dim - kernel_size + 2 * padding + 1
        self.conv_output_size = num_filters * self.output_length

        self.fc = nn.Linear(self.conv_output_size, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = x.flatten(start_dim=1)
        out = self.fc(x)
        return out.squeeze()


def objective(trial, model_name, train_loader, val_loader, device):

    params = {
        'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
        'epochs': trial.suggest_int('epochs', 50, 300)
    }

    if model_name == 'MLP':

        num_layers = trial.suggest_int('mlp_layers', 1, 3)
        hidden_layers = [
            trial.suggest_int(f'mlp_layer_{i}', 32, 256)
            for i in range(num_layers)
        ]

        model_params = {
            'input_dim': 34,
            'hidden_layers': hidden_layers,
            'dropout_rate': trial.suggest_float('mlp_dropout', 0.1, 0.5),
            'activation': nn.ReLU(),
        }
        model = MLP(**model_params)

    elif model_name == 'CNN':
        input_dim = 34
        kernel_size = trial.suggest_int('kernel_size', 3, min(7, input_dim - 1))
        model_params = {
            'num_filters': trial.suggest_int('num_filters', 16, 128),
            'kernel_size': kernel_size,
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            'input_dim': input_dim,
            'padding': 1
        }
        model = CNNModel(**model_params)
    elif model_name == 'LSTM':
        num_layers = trial.suggest_int('lstm_layers', 1, 3)
        hidden_dims = []
        for i in range(num_layers):
            dim = trial.suggest_int(f'lstm_layer_{i}', 32, 256)
            hidden_dims.append(dim)

        model_params = {
            'input_dim': 34,
            'hidden_dims': hidden_dims,
            'dropout': trial.suggest_float('lstm_dropout', 0.1, 0.5)
        }
        model = LSTMModel(**model_params)


    trained_model, _, _, _, _, _ = train_model(
        model,
        train_loader,
        val_loader,
        epochs=params['epochs'],
        lr=params['lr'],
        device=device,
        trial=trial,
        patience=10,
        label_scaler=label_scaler
    )
    r2, rmse, val_mae, mse, _, _ = evaluate_model(
        trained_model,
        val_loader,
        device,
        dataset='val',
        label_scaler=label_scaler
    )
    return val_mae


def evaluate_model(model, test_loader, device="cpu", dataset='val',png_name='model', label_scaler=None):
    model.to(device)
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for batch in test_loader:
            features, labels = batch
            features = features.to(device)
            labels = labels.to(device)

            if isinstance(model, LSTMModel) and features.dim() == 2:
                features = features.unsqueeze(1)

            outputs = model(features)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(outputs.cpu().numpy())

    if label_scaler is not None:
        y_true = label_scaler.inverse_transform(np.array(y_true).reshape(-1, 1)).flatten()
        y_pred = label_scaler.inverse_transform(np.array(y_pred).reshape(-1, 1)).flatten()

    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)

    print(f"{dataset} Set Results - R²: {r2:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f} | MSE: {mse:.4f}")



    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    df.to_csv(f"{current_dir}\\csvs\\{png_name}_{dataset}_predictions.csv", index=False)

    return r2, rmse, mae, mse, y_true, y_pred


class SklearnWrapper:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def fit(self, X, y):
        return self

    def predict(self, X):
        with torch.no_grad():

            if isinstance(self.model, LSTMModel):
                X = X.reshape(X.shape, 1, X.shape)

            elif isinstance(self.model, CNNModel):
                X = X.reshape(X.shape, 1, X.shape)
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            return self.model(X_tensor).cpu().numpy().flatten()

    def score(self, X, y):
        y_pred = self.predict(X)
        return r2_score(y, y_pred)



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def train_model(model, train_loader, val_loader, epochs=50, lr=0.001, device="cpu", patience=10, trial=None, label_scaler=None):

    model.to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_weights = None

    train_losses = []
    val_losses = []
    r2_scores = []
    rmse_scores = []
    mae_scores = []

    for epoch in range(epochs):

        model.train()
        epoch_train_loss = 0.0
        for batch in train_loader:
            features, labels = batch
            features = features.to(device)
            labels = labels.to(device)

            if isinstance(model, LSTMModel) and features.dim() == 2:
                features = features.unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        val_r2, val_rmse, val_mae, val_mse, val_true, val_pred = evaluate_model(
            model,
            val_loader,
            device=device,
            dataset='val',
            label_scaler=label_scaler
        )

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                features, labels = batch
                features = features.to(device)
                labels = labels.to(device)

                if isinstance(model, LSTMModel) and features.dim() == 2:
                    features = features.unsqueeze(1)

                outputs = model(features)
                val_loss += criterion(outputs, labels).item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        r2_scores.append(val_r2)
        rmse_scores.append(val_rmse)
        mae_scores.append(val_mae)

        if avg_val_loss < best_val_loss - 1e-4:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_weights = model.state_dict().copy()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}/{epochs}")
                break

        if trial is not None:
            trial.report(avg_val_loss, epoch)
            if trial.should_prune():
                print(f"Trial pruned at epoch {epoch + 1}")
                raise optuna.exceptions.TrialPruned()

    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
    else:
        print("Warning: No improvement during training, returning last model")

    return model, train_losses, val_losses, r2_scores, rmse_scores, mae_scores

def run_initial_model(model_name, train_loader, val_loader, device):
    initial_params = {
        'lr': 0.001,
        'epochs': 100,
        'mlp_layers': 2,
        'mlp_layer_0': 64,
        'mlp_layer_1': 32,
        'mlp_dropout': 0.3,
    }

    if model_name == 'MLP':
        model = MLP(
            input_dim=34,
            hidden_layers=[initial_params['mlp_layer_0'], initial_params['mlp_layer_1']],
            dropout_rate=initial_params['mlp_dropout'],
            activation=nn.ReLU()
        )
    elif model_name == 'CNN':
        model = CNNModel(
            input_dim=34,
            num_filters=64,
            kernel_size=5,
            dropout=0.3,
            padding=1
        )
    elif model_name == 'LSTM':
        model = LSTMModel(
            input_dim=34,
            hidden_dims=[64],
            dropout=0.3
        )

    model.to(device)

    trained_model, _, _, r2_scores, rmse_scores, mae_scores = train_model(
        model,
        train_loader,
        val_loader,
        epochs=initial_params['epochs'],
        lr=initial_params['lr'],
        device=device
    )

    r2, rmse, mae, mse, y_true, y_pred = evaluate_model(trained_model, val_loader, device, 'val')

    print(f"Initial model performance - R²: {r2:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f} | MSE: {mse:.4f}")

    return r2, rmse, mae, mse


if __name__ == "__main__":

    set_seed(42)
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)

    print("Please select the operation mode:")
    print("1. Carry out Bayesian optimization")
    print("2. Train the model using the optimal parameters")
    operation = input("Please enter a number (1 or 2):").strip()

    print("\nPlease select the model to be used:")
    print("1. MLP")
    print("2. CNN")
    print("3. LSTM")
    model_choice = input("Please enter a number (1-3)：").strip()
    model_name = {
        '1': 'MLP',
        '2': 'CNN',
        '3': 'LSTM'
    }.get(model_choice, 'MLP')
    print(f"\nThe model you have selected is:{model_name} model\n")

    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    try:
        file_path = f"{current_dir}\\datas\\training_data.xlsx"
        index, features, output = read_excel_data(file_path, use_header=True)

        train_loader, val_loader, test_loader, feat_scaler, label_scaler = prepare_dataloaders(
            features=features,
            labels=output,
            test_size=0.1,
            batch_size=32,
            shuffle=True
        )
    except Exception as e:
        print(f"Data loading failed: {str(e)}")
        exit()

    initial_results = run_initial_model(model_name, train_loader, val_loader, device)
    print(
        f"Initial model results: R²: {initial_results[0]:.4f} | RMSE: {initial_results[1]:.4f} | MAE: {initial_results[2]:.4f} | MSE: {initial_results[3]:.4f}")

    if operation == '1':

        try:
            from optuna.samplers import TPESampler

            sampler = TPESampler(seed=42)
            study = optuna.create_study(
                direction='minimize',
                sampler=sampler,
                pruner=optuna.pruners.HyperbandPruner(
                    min_resource=10,
                    max_resource=300,
                    reduction_factor=3
                )
            )
            study.optimize(lambda trial: objective(trial, model_name, train_loader, val_loader, device),
                           n_trials=1,
                           n_jobs=1)

            best_params_dir = os.path.join(current_dir, "best_params")
            os.makedirs(best_params_dir, exist_ok=True)
            best_params_path = os.path.join(best_params_dir, f"{model_name}_best_params.json")
            with open(best_params_path, 'w') as f:
                json.dump(study.best_params, f, indent=4)
            print(f"\nThe optimal parameters have been saved as: {best_params_path}")

        except KeyboardInterrupt:
            print("\nThe user interrupts the optimization process.")
            exit()

    elif operation == '2':
        best_params_dir = os.path.join(current_dir, "best_params")
        best_params_path = os.path.join(best_params_dir, f"{model_name}_best_params.json")

        if not os.path.exists(best_params_path):
            print(f"Error: The saving parameters for {model_name} cannot be found. Please perform Bayesian optimization first!")
            exit()

        with open(best_params_path, 'r') as f:
            best_params = json.load(f)

        print("\n[Load the best parameters for saving]")
        for key, value in best_params.items():
            print(f"  {key}: {value}")

    else:
        print("Invalid operation selection!")
        exit()


    def build_model_from_params(model_name, params):
        try:
            if model_name == 'MLP':
                num_layers = params['mlp_layers']
                hidden_layers = [params[f'mlp_layer_{i}'] for i in range(num_layers)]
                return MLP(
                    input_dim=34,
                    hidden_layers=hidden_layers,
                    dropout_rate=params['mlp_dropout'],
                    activation=nn.ReLU()
                )
            elif model_name == 'CNN':
                return CNNModel(
                    input_dim=34,
                    num_filters=params['num_filters'],
                    kernel_size=params['kernel_size'],
                    dropout=params['dropout'],
                    padding=1
                )
            elif model_name == 'LSTM':
                hidden_dims = [params[f'lstm_layer_{i}'] for i in range(params['lstm_layers'])]
                return LSTMModel(
                    input_dim=34,
                    hidden_dims=hidden_dims,
                    dropout=params['lstm_dropout']
                )
        except KeyError as e:
            print(f"Parameter error: Missing critical parameters {str(e)}")
            exit()


    final_model = build_model_from_params(model_name, study.best_params if operation == '1' else best_params)
    final_model.to(device)

    def merge_dataloaders(train_loader, val_loader):
        merged_features = []
        merged_labels = []
        for batch in train_loader:
            features, labels = batch
            merged_features.append(features)
            merged_labels.append(labels)
        for batch in val_loader:
            features, labels = batch
            merged_features.append(features)
            merged_labels.append(labels)
        merged_features = torch.cat(merged_features, dim=0)
        merged_labels = torch.cat(merged_labels, dim=0)
        return DataLoader(
            TensorDataset(merged_features, merged_labels),
            batch_size=train_loader.batch_size,
            shuffle=True
        )


    merged_train_loader = merge_dataloaders(train_loader, val_loader)

    print("\n[Final Model Training]")
    try:
        trained_model, train_losses, val_losses, r2_scores, rmse_scores, mae_scores = train_model(
            final_model,
            merged_train_loader,
            test_loader,
            epochs=int((study.best_params if operation == '1' else best_params)['epochs'] * 1.2),
            lr=(study.best_params if operation == '1' else best_params)['lr'],
            device=device,
            patience=20,
            label_scaler=label_scaler
        )
    except RuntimeError as e:
        print(f"Training failed: {str(e)}")
        exit()

    print("\n[Visualization of Training Process]")
    try:
        os.makedirs(f'{current_dir}/figures/{model_name}', exist_ok=True)

        plot_training_curves(
            train_losses,
            val_losses,
            r2_scores,
            rmse_scores,
            mae_scores
        )
        print(f"The training curve has been saved as: {current_dir}\\figures\\{model_name}\\{model_name}_train_val_fig.png")
    except Exception as e:
        print(f"Failed to draw the training curve:{str(e)}")

    print("\n[Model Evaluation]")
    evaluation_metrics = {}

    results = {
        'train': {'true': [], 'pred': [], 'metrics': None},
        'val': {'true': [], 'pred': [], 'metrics': None},
        'test': {'true': [], 'pred': [], 'metrics': None}
    }

    for loader, name in [(merged_train_loader, 'train'), (test_loader, 'test')]:
        try:
            r2, rmse, mae, mse, y_true, y_pred = evaluate_model(trained_model,loader,device,dataset=name,png_name='model',label_scaler=label_scaler )
            results[name]['true'] = y_true
            results[name]['pred'] = y_pred
            results[name]['metrics'] = {
                'R²': r2,
                'RMSE': rmse,
                'MAE': mae,
                'MSE': mse
            }
            plot_regression_scatter(y_true, y_pred, model_name, name, results[name]['metrics'])

        except Exception as e:
            print(f"{name} Group assessment failed: {str(e)}")

    train_val_true = np.concatenate([results['train']['true'], results['val']['true']])
    train_val_pred = np.concatenate([results['train']['pred'], results['val']['pred']])

    try:
        print("\n[Comparison Chart of Training Set and Test Set Generation]")
        plot_train_vs_test_scatter(
            y_train_true=results['train']['true'],
            y_train_pred=results['train']['pred'],
            y_test_true=results['test']['true'],
            y_test_pred=results['test']['pred'],
            model_name=model_name,
            train_metrics=results['train']['metrics'],
            test_metrics=results['test']['metrics'],
            current_dir=current_dir
        )
    except Exception as e:
        print(f"Failed to generate the comparison chart: {str(e)}")

    print("\n[Error Distribution Analysis]")
    try:
        plot_combined_error_distribution(
            results['train']['true'], results['train']['pred'],
            results['val']['true'], results['val']['pred'],
            results['test']['true'], results['test']['pred'],
            model_name,
            current_dir
        )
    except Exception as e:
        print(f"Error in generating the error distribution chart: {str(e)}")

    print("\n[Analysis of Importance Ranking]")
    feature_names = features.columns.tolist()
    important_features, importance_scores = compute_permutation_importance(
        trained_model,
        test_loader,
        device,
        feature_names,
        n_repeats=5
    )
    plot_permutation_importance(important_features, importance_scores, model_name, current_dir)

    try:

        pred_file = "prediction_data.xlsx"
        pred_index, pred_features_orig, pred_true_orig = read_excel_data(
            os.path.join(current_dir, "datas", pred_file),
            use_header=True
        )

        X_pred_scaled = feat_scaler.transform(pred_features_orig.values)

        pred_dataset = TensorDataset(
            torch.FloatTensor(X_pred_scaled),
            torch.FloatTensor(pred_true_orig.values)
        )
        pred_loader = DataLoader(pred_dataset, batch_size=32, shuffle=False)

        trained_model.eval()
        pred_values_scaled = []
        with torch.no_grad():
            for features, _ in pred_loader:
                features = features.to(device)

                if isinstance(trained_model, LSTMModel) and features.dim() == 2:
                    features = features.unsqueeze(1)
                outputs = trained_model(features)
                pred_values_scaled.append(outputs.cpu().numpy())

        y_pred_scaled = np.concatenate(pred_values_scaled).flatten()
        y_pred_orig = inverse_scale(label_scaler, y_pred_scaled)

        y_true_orig = pred_true_orig.values.flatten()

        df_orig = pd.DataFrame({
            "True_Original": y_true_orig,
            "Predicted_Original": y_pred_orig
        })
        csv_path = os.path.join(current_dir, "csvs",
                                f"{model_name}_pred_original_scale.csv")
        df_orig.to_csv(csv_path, index=False)
        print(f"Original scale data has been saved:{csv_path}")

        plot_true_vs_prediction(y_true_orig, y_pred_orig,
                                model_name, current_dir)

    except Exception as e:
        print(f"Prediction set processing failed:{str(e)}")
