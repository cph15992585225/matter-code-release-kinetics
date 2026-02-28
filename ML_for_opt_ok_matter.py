import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, ConstantKernel as C
from sklearn.linear_model import Lasso
import shap
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from dataset import prepare_dataloaders, read_excel_data
import os
from sklearn.model_selection import learning_curve, ShuffleSplit, train_test_split
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from sklearn.neighbors import KNeighborsRegressor
from sklearn.inspection import permutation_importance
from scipy.stats import skew

from sklearn.pipeline import Pipeline
import traceback

plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 26,
    'font.weight': 'normal',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'axes.linewidth': 3,
})

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_name, save_pred=False, initial=False):
    if initial:
        model = model.__class__()
        print(f"Initial {model_name} parameters: {model.get_params()}")

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)

    y_test_pred = model.predict(X_test)

    metrics = {}
    for split, y_true, y_pred in zip(["Train", "Test"],
                                     [y_train, y_test],
                                     [y_train_pred, y_test_pred]):
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        metrics[split] = {"R²": r2, "RMSE": rmse, "MAE": mae, "MSE": mse}

        print(f"{split} Results - R²: {r2:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f} | MSE: {mse:.4f}")

        plot_regression_scatter(y_true, y_pred, model_name, split.lower(), metrics[split])

    plot_train_test_scatter(y_train, y_train_pred, y_test, y_test_pred, model_name)

    plot_predictions(y_test, y_test_pred, model_name)
    plot_combined_error_distribution(y_train_true=y_train,y_train_pred=y_train_pred,y_test_true=y_test,y_test_pred=y_test_pred,model_name=model_name)
    import pandas as pd
    if save_pred:
        df = pd.DataFrame({"y_train": np.array(y_train), "y_train_pred": np.array(y_train_pred)})
        y_train_pred = pd.DataFrame(y_train_pred)

        df.to_csv("{}\csvs\\train_mechine_learning_{}_train_predictions.csv".format(current_dir, model_name),
                  index=False)
        df = pd.DataFrame({"y_test": np.array(y_test), "y_test_pred": np.array(y_test_pred)})
        y_test_pred = pd.DataFrame(y_test_pred)

        df.to_csv("{}\csvs\\train_mechine_learning_{}_test_predictions.csv".format(current_dir, model_name),
                  index=False)

        print("测试集的真实和预测数据已成功保存到\csvs\\train_mechine_learning_train/val/test_predictions.csv！".format(
            model_name))

    return model, metrics


def plot_learning_curve(model, X, y, model_name):
    plt.figure(figsize=(10, 8))

    ax1 = plt.gca()
    ax2 = ax1.twinx()

    train_sizes, train_r2, test_r2 = learning_curve(
        estimator=model,
        X=X,
        y=y,
        cv=10,
        scoring='r2',
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        verbose=0
    )

    _, train_mae, test_mae = learning_curve(
        estimator=model,
        X=X,
        y=y,
        cv=10,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        verbose=0
    )

    train_mae = -train_mae
    test_mae = -test_mae

    def get_stats(scores):
        return np.mean(scores, axis=1), np.std(scores, axis=1)

    train_r2_mean, train_r2_std = get_stats(train_r2)
    test_r2_mean, test_r2_std = get_stats(test_r2)
    train_mae_mean, train_mae_std = get_stats(train_mae)
    test_mae_mean, test_mae_std = get_stats(test_mae)

    (train_r2_line,) = ax1.plot(train_sizes, train_r2_mean,
                                color='#1f77b4', marker='o', linestyle='-',
                                markersize=16, linewidth=3, label='Train R²')
    (test_r2_line,) = ax1.plot(train_sizes, test_r2_mean,
                               color='#ff7f0e', marker='s', linestyle='-',
                               markersize=16, linewidth=3, label='Val R²')
    (train_mae_line,) = ax2.plot(train_sizes, train_mae_mean,
                                 color='#2ca02c', marker='^', linestyle='--',
                                 markersize=16, linewidth=3, label='Train MAE')
    (test_mae_line,) = ax2.plot(train_sizes, test_mae_mean,
                                color='#d62728', marker='D', linestyle='--',
                                markersize=16, linewidth=3, label='Val MAE')

    alpha = 0.15
    ax1.fill_between(train_sizes,
                     train_r2_mean - train_r2_std,
                     train_r2_mean + train_r2_std,
                     alpha=alpha, color='#1f77b4')
    ax1.fill_between(train_sizes,
                     test_r2_mean - test_r2_std,
                     test_r2_mean + test_r2_std,
                     alpha=alpha, color='#ff7f0e')
    ax2.fill_between(train_sizes,
                     train_mae_mean - train_mae_std,
                     train_mae_mean + train_mae_std,
                     alpha=alpha, color='#2ca02c')
    ax2.fill_between(train_sizes,
                     test_mae_mean - test_mae_std,
                     test_mae_mean + test_mae_std,
                     alpha=alpha, color='#d62728')

    ax1.set_xlabel("Training Examples", fontsize=28, labelpad=10)
    ax1.set_ylabel("R² Score", fontsize=28, labelpad=10)
    ax2.set_ylabel("MAE", fontsize=28, labelpad=10, rotation=270, va='bottom')
    ax1.tick_params(axis='both', labelsize=26, which='major')
    ax2.tick_params(axis='y', labelsize=26)

    mae_min = min(np.min(train_mae_mean - train_mae_std),
                  np.min(test_mae_mean - test_mae_std))
    mae_max = max(np.max(train_mae_mean + train_mae_std),
                  np.max(test_mae_mean + test_mae_std))

    #padding = (mae_max - mae_min) * 0.05
    #ax2.set_ylim(max(0, mae_min - padding),  mae_max *2)
    ax2.set_ylim(0, 8)
    ax1.set_ylim(0.0, 1.1)

    lines = [train_r2_line, test_r2_line, train_mae_line, test_mae_line]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels,
               loc='center right',
               bbox_to_anchor=(0.95, 0.4),
               fontsize=26,
               frameon=False,
               facecolor='white')

    ax1.grid(False)
    plt.title(f'Dual Metric Learning Curve - {model_name}',
              fontsize=26, pad=20, fontweight='bold')

    save_dir = os.path.join(current_dir, 'figures', model_name)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{model_name}_dual_learning_curve.tiff"),
                dpi=600, format='tiff')
    plt.close()
    print(f"The dual-index learning curve has been saved to: {save_dir}{model_name}_dual_learning_curve.png")



def plot_regression_scatter(y_true, y_pred, model_name, dataset_type, metrics):


    plt.figure(figsize=(10, 8))
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

    ax.set_xlabel('True Value', fontsize=26)
    ax.set_ylabel('Predicted Value', fontsize=26)
    ax.set_title(model_name, fontsize=26)
    ax.tick_params(axis='both', labelsize=26, width=1.5)
    ax.grid(False)
    ax.legend().set_visible(False)

    metrics_text = (
        f"{dataset_type.capitalize()} Metrics:\n"
        f"R² = {metrics['R²']:.4f}\n"
        f"RMSE = {metrics['RMSE']:.4f}\n"
        f"MSE = {metrics['MSE']:.4f}\n"
        f"MAE = {metrics['MAE']:.4f}"
    )
    plt.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
             fontsize=26, fontweight='bold', verticalalignment='top',
             horizontalalignment='left',
             )

    save_dir = os.path.join(current_dir, 'figures', model_name)
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f'{model_name}_{dataset_type}_scatter.tiff')

    plt.savefig(save_path, bbox_inches='tight', dpi=600, format='tiff')
    plt.close()
    print(f"Saved to: {save_path}")


def plot_feature_contribution_vs_r2(model, X_train, y_train, X_val, y_val):

    result = permutation_importance(model, X_train, y_train, n_repeats=10, random_state=42, n_jobs=-1)

    feature_contributions = result.importances_mean

    y_val_pred = model.predict(X_val)
    r2 = r2_score(y_val, y_val_pred)

    plt.figure(figsize=(10, 8))
    plt.barh(range(len(feature_contributions)), feature_contributions, align='center')
    plt.yticks(range(len(feature_contributions)), X_train.columns)
    plt.xlabel('Feature Contribution')
    plt.title(f'Feature Contribution vs R² (R² = {r2:.4f})')
    plt.tight_layout()
    plt.savefig(f"{current_dir}/figures/feature_contribution_vs_r2.png")
    plt.show()
    print(f"Feature contribution vs R² plot saved.")


def plot_combined_error_distribution(y_train_true, y_train_pred,y_test_true, y_test_pred,model_name):

    errors = {
        'Train': y_train_true - y_train_pred,

        'Test': y_test_true - y_test_pred
    }

    fig, ax1 = plt.subplots(figsize=(10, 8))
    ax2 = ax1.twinx()
    plt.rcParams['font.family'] = 'Times New Roman'

    color_palette = {
        'Train': ('#1f77b4', '#6A9ACE'),
        'Validation': ('#2ca02c', '#A6DBA0'),
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
                     label=f'{dataset} ')

    for dataset, err in errors.items():
        kde_color, hist_color = color_palette[dataset]

        kde_plot = sns.kdeplot(err,
                               ax=ax2,
                               color=kde_color,
                               linewidth=2.5,
                               label=f'{dataset} KDE')
        kde_lines.append(kde_plot.get_lines()[-1])

    ax1.set_xlabel("Prediction Error", fontsize=28, labelpad=10)
    ax1.set_ylabel("Frequency", fontsize=28, labelpad=10)
    ax2.set_ylabel("Density", fontsize=28, labelpad=10, rotation=270, va='bottom')

    x_min = min(np.concatenate([err for err in errors.values()]))
    x_max = max(np.concatenate([err for err in errors.values()]))
    ax1.set_xlim(x_min * 1.1, x_max * 1.1)
    ax2.spines['right'].set_position(('axes', 1))
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2 = kde_lines
    labels2 = [l.get_label() for l in handles2]
    all_handles = handles1 + handles2
    all_labels = labels1 + labels2
    seen = set()
    unique_handles = []
    unique_labels = []
    for h, l in zip(all_handles, all_labels):
        if l not in seen:
            seen.add(l)
            unique_handles.append(h)
            unique_labels.append(l)
    ax1.legend(unique_handles, unique_labels,
               loc='upper left',
               fontsize=26,
               title='Dataset & Type',
               title_fontsize=26,
               frameon=False,
               framealpha=0.9)
    stats_text = ""
    for dataset, err in errors.items():
        stats_text += (
            f"{dataset}:\n"
            f"μ = {err.mean():.3f}\n"
            f"σ = {err.std():.3f}\n"
            f"Skew = {skew(err):.2f}\n\n"
        )

    ax1.text(0.72, 0.97, stats_text,
             transform=ax1.transAxes,
             ha='left',
             va='top',
             fontsize=26,
             bbox=dict(facecolor='none', alpha=0,
                       edgecolor='gray', boxstyle='round'))

    plt.title(f"Dual-Axis Error Analysis - {model_name}",
              fontsize=26, pad=20)

    ax1.tick_params(axis='both', which='major', labelsize=26)
    ax2.tick_params(axis='y', labelsize=26)
    save_dir = os.path.join(current_dir, 'figures', model_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{model_name}_dual_axis_error_dist.tiff')
    plt.savefig(save_path, bbox_inches='tight', dpi=600, format='tiff')
    plt.close()
    print(f"Dual-axis error distribution saved: {save_path}")



def plot_train_test_scatter(y_train_true, y_train_pred, y_test_true, y_test_pred, model_name):

    plt.figure(figsize=(10, 8))
    ax = plt.gca()

    plt.scatter(y_train_true, y_train_pred, alpha=0.8,
                edgecolor='#6A9ACE', facecolor='none', marker='o', s=100, linewidths=1.5)

    plt.scatter(y_test_true, y_test_pred, alpha=0.8,
                edgecolor='#F1766D', facecolor='none', marker='o', s=100, linewidths=1.5)

    slope_train, intercept_train = np.polyfit(y_train_true, y_train_pred, 1)
    reg_line_train = slope_train * y_train_true + intercept_train
    r2_train = r2_score(y_train_true, y_train_pred)
    rmse_train = np.sqrt(mean_squared_error(y_train_true, y_train_pred))
    mse_train = mean_squared_error(y_train_true, y_train_pred)
    mae_train = mean_absolute_error(y_train_true, y_train_pred)

    slope_test, intercept_test = np.polyfit(y_test_true, y_test_pred, 1)
    reg_line_test = slope_test * y_test_true + intercept_test
    r2_test = r2_score(y_test_true, y_test_pred)
    rmse_test = np.sqrt(mean_squared_error(y_test_true, y_test_pred))
    mse_test = mean_squared_error(y_test_true, y_test_pred)
    mae_test = mean_absolute_error(y_test_true, y_test_pred)

    plt.plot(y_train_true, reg_line_train, color='#4A7DA6',
             )

    plt.plot(y_test_true, reg_line_test, color='#C25A53',
             )

    plt.plot([min(y_train_true.min(), y_test_true.min()), max(y_train_true.max(), y_test_true.max())],
             [min(y_train_true.min(), y_test_true.min()), max(y_train_true.max(), y_test_true.max())],
             'k--', lw=2)

    ax.set_xlabel('True Value', fontsize=26)
    ax.set_ylabel('Predicted Value', fontsize=26)
    ax.set_title(f'{model_name} ', fontsize=26)
    ax.grid(False)
    ax.tick_params(axis='x', labelsize=26)
    ax.tick_params(axis='y', labelsize=26)

    train_metrics_text = (
        f"Train Metrics:\n"
        f"R² = {r2_train:.4f}\n"
        f"MSE = {mse_train:.4f}\n"
        f"RMSE = {rmse_train:.4f}\n"
        f"MAE = {mae_train:.4f}"
    )
    plt.text(0.05, 0.95, train_metrics_text, transform=ax.transAxes,
             fontsize=26, fontweight='bold', verticalalignment='top', color='#6A9ACE',
             horizontalalignment='left')
    test_metrics_text = (
        f"Test Metrics:\n"
        f"R² = {r2_test:.4f}\n"
        f"MSE = {mse_test:.4f}\n"
        f"RMSE = {rmse_test:.4f}\n"
        f"MAE = {mae_test:.4f}"
    )
    plt.text(0.95, 0.05, test_metrics_text, transform=ax.transAxes,
             fontsize=26, fontweight='bold', verticalalignment='bottom', color='#F1766D',
             horizontalalignment='right')

    ax.legend(loc='upper left', bbox_to_anchor=(0.05, 0.95), frameon=False)
    ax.legend(loc='lower right', bbox_to_anchor=(0.95, 0.05), frameon=False)

    save_dir = os.path.join(current_dir, 'figures', model_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{model_name}_train_test_scatter.tiff')
    plt.savefig(save_path, bbox_inches='tight', dpi=600, format='tiff')
    plt.close()
    print(f"Train vs Test set scatter plot saved to: {save_path}")



def plot_predictions(y_true, y_pred, model_name):

    indices = np.arange(len(y_true))

    plt.figure(figsize=(10, 8))
    plt.plot(indices, y_true, label="True Values", color='blue', linestyle='-', marker='o', markersize=16)
    plt.plot(indices, y_pred, label="Predicted Values", color='red', linestyle='-', marker='s', markersize=16)

    plt.xlabel("Sample Index", fontsize=26)
    plt.ylabel("Values", fontsize=26)
    plt.title("True vs Predicted Values", fontsize=26)
    plt.legend()
    plt.grid(False)
    save_dir = os.path.join(current_dir, 'figures', model_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{model_name}_predictions.tiff')
    plt.savefig(save_path, dpi=600, format='tiff')
    plt.close()


def preprocess_data(features_all, output_all, test_size=0.1, random_state=42):
    np.random.seed(random_state)
    shuffled_indices = np.random.permutation(len(features_all))
    features_shuffled = features_all.iloc[shuffled_indices]
    output_shuffled = output_all.iloc[shuffled_indices]

    X_train, X_test, y_train, y_test = train_test_split(
        features_shuffled, output_shuffled,
        test_size=test_size, random_state=random_state
    )
    return X_train, y_train, X_test, y_test


def plot_feature_importance(model, X_train, y_train, feature_names=None, n_top=15):
    if isinstance(model, Pipeline):
        model = model.named_steps['model']
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        importances = importances / importances.sum()
    elif isinstance(model, SVR):
        result = permutation_importance(model, X_train, y_train, n_repeats=10, random_state=42, n_jobs=-1)
        importances = result.importances_mean
    elif isinstance(model, LinearRegression):
        importances = np.abs(model.coef_)
    elif isinstance(model, GaussianProcessRegressor):
        result = permutation_importance(
            model, X_train, y_train,
            n_repeats=10, random_state=42, n_jobs=-1
        )
        importances = result.importances_mean
    elif isinstance(model, KNeighborsRegressor):
        result = permutation_importance(model, X_train, y_train, n_repeats=10, random_state=42, n_jobs=-1)
        importances = result.importances_mean
    elif isinstance(model, Lasso):
        importances = np.abs(model.coef_)
        if np.all(importances == 0):
            importances += 1e-9
    else:
        raise ValueError("This model does not support the calculation of feature importance!")

    if feature_names is None:
        if isinstance(X_train, pd.DataFrame):
            feature_names = X_train.columns
        else:
            feature_names = [f"F{i}" for i in range(X_train.shape)]

    indices = np.argsort(importances)[::-1]

    df = pd.DataFrame({"indices": indices, "Score": importances[indices]})
    df.to_csv(f"{current_dir}/csvs/train_mechine_learning_{model_name}_indices.csv", index=False)

    plt.figure(figsize=(10, 8))
    plt.barh(
        y=range(n_top),
        width=importances[indices[:n_top]],
        align='center',
        color='#084594'
    )

    # 坐标轴调整
    plt.yticks(
        ticks=range(n_top),
        labels=[feature_names[i] for i in indices[:n_top]]
    )
    plt.gca().invert_yaxis()
    plt.xticks(fontsize=26)
    plt.xlabel("Importance Score",fontsize=26)

    plt.title("Feature Importance",fontsize=26)

    plt.grid(False)
    plt.tight_layout()

    save_dir = os.path.join(current_dir, 'figures', model_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{model_name}_feature_importance.tiff')
    plt.savefig(save_path, dpi=600, format='tiff')
    plt.close()

import matplotlib.ticker as ticker
def predict_and_plot_lines(model, model_name, current_dir):

    pred_path = os.path.join(current_dir, "datas", "prediction_data.xlsx")
    _, features_pred, output_pred = read_excel_data(pred_path, use_header=True)

    y_pred = model.predict(features_pred.values)
    y_true = output_pred.values.flatten()

    pred_df = pd.DataFrame({
        "Sample_Index": range(len(y_true)),
        "True_Value": y_true,
        "Predicted_Value": y_pred
    })
    os.makedirs(os.path.join(current_dir, "csvs"), exist_ok=True)
    pred_df.to_csv(os.path.join(current_dir, "csvs", f"{model_name}_prediction_results.csv"), index=False)

    def calculate_f2(true_values, pred_values):
        n = len(true_values)
        sum_sq_diff = np.sum((np.array(true_values) - np.array(pred_values)) ** 2)
        denominator = np.sqrt(1 + sum_sq_diff / n) + 1e-10
        return 50 * np.log10(100 / denominator)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 18,
        'axes.labelpad': 10
    })

    split_idx = 16

    true_first, pred_first = y_true[:split_idx], y_pred[:split_idx]
    true_remain, pred_remain = y_true[split_idx:], y_pred[split_idx:]


    ax1.plot(pred_df["Sample_Index"][:split_idx], true_first,
             's-', color='#1f77b4', markersize=16, linewidth=2, label='True Values')
    ax1.plot(pred_df["Sample_Index"][:split_idx], pred_first,
             'o--', color='#d62728', markersize=16, linewidth=2, markerfacecolor='none',
             markeredgewidth=2, label='Predicted Values')

    for idx in range(split_idx):
        ax1.plot([idx, idx], [true_first[idx], pred_first[idx]],
                 color='gray', linestyle=':', alpha=0.9)

    r2 = r2_score(true_first, pred_first)
    rmse = np.sqrt(mean_squared_error(true_first, pred_first))
    f2 = calculate_f2(true_first, pred_first)
    mse = mean_squared_error(true_first, pred_first)

    x_left = pred_df["Sample_Index"][:split_idx].values
    area_left = np.trapz(np.abs(true_first - pred_first), x=x_left)

    metrics_text = (f"R² = {r2:.3f}\nRMSE = {rmse:.3f}\n"
                    f"MAE = {mean_absolute_error(true_first, pred_first):.3f}\n"
                    f"MSE = {mse:.1f}\n"
                    f"f₂ = {f2:.1f}\n"
                    f"PTDA = {area_left:.1f}")

    ax1.text(0.7, 0.12, metrics_text, transform=ax1.transAxes,
             fontsize=26)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(3))
    ax1.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{x / 6:.1f}")
    )
    ax1.set_title(f"{model_name} Prediction Comparison (Model I)", fontsize=26, pad=12)
    ax1.set_xlabel("Time (h)", fontsize=28)
    ax1.set_ylabel("Q (%)", fontsize=28)
    ax1.tick_params(axis='both', labelsize=26)
    ax1.legend(fontsize=26, loc='upper left')
    ax1.legend(frameon=False)

    ax2.plot(pred_df["Sample_Index"][split_idx:], true_remain,
             's-', color='#1f77b4', markersize=16, linewidth=2, label='True Values')
    ax2.plot(pred_df["Sample_Index"][split_idx:], pred_remain,
             'o--', color='#d62728', markersize=16, linewidth=2, markerfacecolor='none',
             markeredgewidth=2, label='Predicted Values')

    for idx in range(len(true_remain)):
        ax2.plot([idx + split_idx, idx + split_idx],
                 [true_remain[idx], pred_remain[idx]],
                 color='gray', linestyle=':', alpha=0.9)

    r2_r = r2_score(true_remain, pred_remain)
    rmse_r = np.sqrt(mean_squared_error(true_remain, pred_remain))
    f2_r = calculate_f2(true_remain, pred_remain)
    mse = mean_squared_error(true_remain, pred_remain)

    x_right = pred_df["Sample_Index"][split_idx:].values
    area_right = np.trapz(np.abs(true_remain - pred_remain), x=x_right)

    metrics_text_r = (f"R² = {r2_r:.3f}\nRMSE = {rmse_r:.3f}\n"
                      f"MAE = {mean_absolute_error(true_remain, pred_remain):.3f}\n"
                      f"MSE = {mse:.1f}\n"
                      f"f₂ = {f2_r:.1f}\n"
                      f"PTDA = {area_right:.1f}")

    ax2.text(0.7, 0.12, metrics_text_r, transform=ax2.transAxes,
             fontsize=26)
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(12))
    ax2.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{x / 6:.0f}")
    )
    ax2.set_title(f"{model_name} Prediction Comparison (Model II)", fontsize=26, pad=12)
    ax2.set_xlabel("Time (h)", fontsize=28)
    ax2.set_ylabel("Q (%)", fontsize=28)
    ax2.tick_params(axis='both', labelsize=26)
    ax2.legend(fontsize=26, loc='upper left')
    ax2.legend(frameon=False)
    save_dir = os.path.join(current_dir, 'figures', model_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{model_name}_prediction_lines.tiff")
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, format='tiff', bbox_inches='tight')
    plt.close()
    print(f"The point-line comparison chart has been saved as follows: {save_path}")



def reference_and_plot_lines(model, model_name, current_dir):

    pred_path = os.path.join(current_dir, "datas", "training_data.xlsx")
    _, features_pred, output_pred = read_excel_data(pred_path, use_header=True)

    y_pred = model.predict(features_pred.values)
    y_true = output_pred.values.flatten()


    pred_df = pd.DataFrame({
        "Sample_Index": range(len(y_true)),
        "True_Value": y_true,
        "Predicted_Value": y_pred
    })
    os.makedirs(os.path.join(current_dir, "csvs"), exist_ok=True)
    pred_df.to_csv(os.path.join(current_dir, "csvs", f"{model_name}_5mg优化_results.csv"), index=False)

    def calculate_f2(true_values, pred_values):
        n = len(true_values)
        sum_sq_diff = np.sum((np.array(true_values) - np.array(pred_values)) ** 2)
        denominator = np.sqrt(1 + sum_sq_diff / n) + 1e-10
        return 50 * np.log10(100 / denominator)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 18,
        'axes.labelpad': 10
    })

    split_idx = 79
    true_first, pred_first = y_true[:split_idx], y_pred[:split_idx]
    true_remain, pred_remain = y_true[split_idx:], y_pred[split_idx:]


    ax1.plot(pred_df["Sample_Index"][:split_idx], true_first,
             's-', color='#1f77b4', markersize=16, linewidth=2, label='True Values')
    ax1.plot(pred_df["Sample_Index"][:split_idx], pred_first,
             'o--', color='#d62728', markersize=16, linewidth=2, markerfacecolor='none',
             markeredgewidth=2, label='Predicted Values')

    for idx in range(split_idx):
        ax1.plot([idx, idx], [true_first[idx], pred_first[idx]],
                 color='gray', linestyle=':', alpha=0.9)

    r2 = r2_score(true_first, pred_first)
    rmse = np.sqrt(mean_squared_error(true_first, pred_first))
    f2 = calculate_f2(true_first, pred_first)
    mse = mean_squared_error(true_first, pred_first)

    x_left = pred_df["Sample_Index"][:split_idx].values
    area_left = np.trapz(np.abs(true_first - pred_first), x=x_left)

    metrics_text = (f"R² = {r2:.3f}\nRMSE = {rmse:.3f}\n"
                    f"MAE = {mean_absolute_error(true_first, pred_first):.3f}\n"
                    f"MSE = {mse:.1f}\n"
                    f"f₂ = {f2:.1f}\n"
                    f"PTDA = {area_left:.1f}")

    ax1.text(0.7, 0.12, metrics_text, transform=ax1.transAxes,
             fontsize=26)

    ax1.xaxis.set_major_locator(ticker.MultipleLocator(3))
    ax1.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{x / 6:.1f}")
    )
    ax1.set_title(f"{model_name} Prediction Comparison (Model I)", fontsize=26, pad=12)
    ax1.set_xlabel("Time (h)", fontsize=28)
    ax1.set_ylabel("Q (%)", fontsize=28)
    ax1.tick_params(axis='both', labelsize=26)
    ax1.legend(fontsize=26, loc='upper left')
    ax1.legend(frameon=False)

    ax2.plot(pred_df["Sample_Index"][split_idx:], true_remain,
             's-', color='#1f77b4', markersize=16, linewidth=2, label='True Values')
    ax2.plot(pred_df["Sample_Index"][split_idx:], pred_remain,
             'o--', color='#d62728', markersize=16, linewidth=2, markerfacecolor='none',
             markeredgewidth=2, label='Predicted Values')

    for idx in range(len(true_remain)):
        ax2.plot([idx + split_idx, idx + split_idx],
                 [true_remain[idx], pred_remain[idx]],
                 color='gray', linestyle=':', alpha=0.9)

    r2_r = r2_score(true_remain, pred_remain)
    rmse_r = np.sqrt(mean_squared_error(true_remain, pred_remain))
    f2_r = calculate_f2(true_remain, pred_remain)
    mse = mean_squared_error(true_remain, pred_remain)

    x_right = pred_df["Sample_Index"][split_idx:].values
    area_right = np.trapz(np.abs(true_remain - pred_remain), x=x_right)

    metrics_text_r = (f"R² = {r2_r:.3f}\nRMSE = {rmse_r:.3f}\n"
                      f"MAE = {mean_absolute_error(true_remain, pred_remain):.3f}\n"
                      f"MSE = {mse:.1f}\n"
                      f"f₂ = {f2_r:.1f}\n"
                      f"PTDA = {area_right:.1f}")

    ax2.text(0.7, 0.12, metrics_text_r, transform=ax2.transAxes,
             fontsize=26)
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(12))
    ax2.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{x / 6:.0f}")
    )
    ax2.set_title(f"{model_name} Prediction Comparison (Model II)", fontsize=26, pad=12)
    ax2.set_xlabel("Time (h)", fontsize=26)
    ax2.set_ylabel("Q (%)", fontsize=26)
    ax2.tick_params(axis='both', labelsize=26)
    ax2.legend(fontsize=26, loc='upper left')
    ax2.legend(frameon=False)

    save_dir = os.path.join(current_dir, 'figures', model_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{model_name}_reference_prediction_lines.tiff")
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, format='tiff', bbox_inches='tight')
    plt.close()
    print(f"The line comparison chart of the reference formulation has been saved at: {save_path}")


def grid_search_model(model, param_space, X_train, y_train):

    bayes_search = BayesSearchCV(model, param_space, n_iter=100, scoring='neg_mean_absolute_error', n_jobs=-1, cv=5, verbose=1)
    bayes_search.fit(X_train, y_train)
    print("Optimal parameters:", bayes_search.best_params_)
    return bayes_search.best_estimator_

def create_gpr_kernel(kernel_name):
    kernel_map = {
        'RBF': C(1.0) * RBF(10.0),
        'Matern': C(1.0) * Matern(nu=1.5),
        'RationalQuadratic': C(1.0) * RationalQuadratic()
    }
    return kernel_map.get(kernel_name, RBF())
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)





def cross_val_stability(model, X, y, cv=5, model_name=""):

    scoring = {
        'r2': 'r2',
        'rmse': 'neg_root_mean_squared_error',
        'mae': 'neg_mean_absolute_error'
    }

    cv_results = cross_validate(
        model, X, y, cv=cv, scoring=scoring,
        return_train_score=True, n_jobs=-1
    )


    metrics = {
        'train_r2': cv_results['train_r2'],
        'test_r2': cv_results['test_r2'],
        'train_rmse': -cv_results['train_rmse'],
        'test_rmse': -cv_results['test_rmse'],
        'train_mae': -cv_results['train_mae'],
        'test_mae': -cv_results['test_mae']
    }

    name_mapping = {
        'train_r2': 'Train R²',
        'test_r2': 'Test R²',
        'train_rmse': 'Train RMSE',
        'test_rmse': 'Test RMSE',
        'train_mae': 'Train MAE',
        'test_mae': 'Test MAE'
    }

    if model_name:
        print(f"\nModel: {model_name}")
    for key in metrics:
        display_name = name_mapping[key]
        mean = np.mean(metrics[key])
        std = np.std(metrics[key], ddof=1)
        print(f"{display_name}: {mean:.3f} ± {std:.3f}")

    plot_cv_stability(metrics, model_name)

    return metrics


def plot_cv_stability(metrics, model_name):

    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'axes.titlesize': 26,
        'axes.labelsize': 26,
        'xtick.labelsize': 26,
        'ytick.labelsize': 26,
        'legend.fontsize': 26,
        'legend.title_fontsize': 26,
        'axes.linewidth': 3,
        'lines.linewidth': 3,
        'lines.markersize': 16
    })


    fig = plt.figure(figsize=(12, 9), dpi=300)
    fig.suptitle(f'{model_name} Cross-Validation Stability', y=0.92, fontsize=26,fontweight='bold' )


    palette = {'train': '#4B72B1', 'test': '#DD8449'}
    line_styles = {'train': '-', 'test': '--'}


    ax1 = plt.subplot(2, 2, 1)

    box = ax1.boxplot([metrics['train_r2'], metrics['test_r2']],
                      widths=0.5, patch_artist=True, whis=1.5,
                      flierprops=dict(marker='o', markersize=16, markerfacecolor='none', markeredgecolor='#555555'))

    for patch, color in zip(box['boxes'], [palette['train'], palette['test']]):
        patch.set_edgecolor(color)
        patch.set_facecolor('none')
        patch.set_linewidth(2)

    for whisker, color in zip(box['whiskers'], [palette['train'], palette['train'], palette['test'], palette['test']]):
        whisker.set_color(color)
        whisker.set_linewidth(2)
    for cap, color in zip(box['caps'], [palette['train'], palette['train'], palette['test'], palette['test']]):
        cap.set_color(color)
        cap.set_linewidth(2)
    for median, color in zip(box['medians'], [palette['train'], palette['test']]):
        median.set_color(color)
        median.set_linewidth(2)

    for i, (data, color) in enumerate(zip([metrics['train_r2'], metrics['test_r2']], [palette['train'], palette['test']])):
        x = np.random.normal(i + 1, 0.04, size=len(data))
        ax1.plot(x, data, 'o', color=color, alpha=0.6, markersize=16)
    ax1.set_xticks([1, 2])
    ax1.set_xticklabels(['Train R²', 'Test R²'])
    ax1.set_ylabel('R² Score', labelpad=10)
    ax1.grid(False)
    ax1.set_ylim(0.994, 1.000)


    ax2 = plt.subplot(2, 2, 2)
    box = ax2.boxplot([metrics['train_rmse'], metrics['test_rmse']],
                      widths=0.5, patch_artist=True, whis=1.5,
                      flierprops=dict(marker='o', markersize=16, markerfacecolor='none', markeredgecolor='#555555'))
    for patch, color in zip(box['boxes'], [palette['train'], palette['test']]):
        patch.set_edgecolor(color)
        patch.set_facecolor('none')
        patch.set_linewidth(2)
    for whisker, color in zip(box['whiskers'], [palette['train'], palette['train'], palette['test'], palette['test']]):
        whisker.set_color(color)
        whisker.set_linewidth(2)
    for cap, color in zip(box['caps'], [palette['train'], palette['train'], palette['test'], palette['test']]):
        cap.set_color(color)
        cap.set_linewidth(2)
    for median, color in zip(box['medians'], [palette['train'], palette['test']]):
        median.set_color(color)
        median.set_linewidth(2)

    for i, (data, color) in enumerate(zip([metrics['train_rmse'], metrics['test_rmse']], [palette['train'], palette['test']])):
        x = np.random.normal(i + 1, 0.04, size=len(data))
        ax2.plot(x, data, 'o', color=color, alpha=0.6, markersize=16)
    ax2.set_xticks([1, 2])
    ax2.set_xticklabels(['Train RMSE', 'Test RMSE'])
    ax2.set_ylabel('RMSE', labelpad=10)
    ax2.grid(False)
    ax2.set_ylim(0.4, 2.1)

    ax3 = plt.subplot(2, 2, 3)
    fold_numbers = np.arange(len(metrics['train_mae'])) + 1
    ax3.plot(fold_numbers, metrics['train_mae'],
             color=palette['train'], linestyle=line_styles['train'],
             marker='o', markersize=16, label='Train')
    ax3.plot(fold_numbers, metrics['test_mae'],
             color=palette['test'], linestyle=line_styles['test'],
             marker='s', markersize=16, label='Test')
    ax3.set_xlabel('Fold Number', labelpad=10)
    ax3.set_ylabel('MAE', labelpad=10)
    ax3.set_xticks(fold_numbers)
    ax3.grid(False)
    ax3.set_ylim(0, 1.6)

    ax4 = plt.subplot(2, 2, 4)
    sc = ax4.scatter(metrics['test_r2'], metrics['test_rmse'],
                     c=fold_numbers, cmap='plasma', s=120,
                     edgecolor='w', linewidth=0.5)
    ax4.set_xlabel('Test R²', labelpad=10)
    ax4.set_ylabel('Test RMSE', labelpad=10)
    cbar = fig.colorbar(sc, ax=ax4)
    cbar.set_label('Fold Number', rotation=270, labelpad=15)


    z = np.polyfit(metrics['test_r2'], metrics['test_rmse'], 1)
    p = np.poly1d(z)

    ax4.plot(metrics['test_r2'], p(metrics['test_r2']),
             linestyle='--',
             dashes=(3,2),
             color='#555555',
             alpha=0.8,
             linewidth=2,
             solid_capstyle='round',
             label=f'ρ = {np.corrcoef(metrics["test_r2"], metrics["test_rmse"])[0, 1]:.2f}')
    ax4.legend(frameon=False)
    ax4.grid(False)


    plt.tight_layout(rect=[0, 0, 1, 0.96], h_pad=1, w_pad=1)


    save_dir = os.path.join(current_dir, 'figures', model_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{model_name}_cv_stability.tiff")
    plt.savefig(save_path, dpi=600, format='tiff', bbox_inches='tight')
    plt.close(fig)


def perform_shap_analysis(model, X_train, X_test, feature_names, model_name, current_dir, n_features=15,
                         sample_num=40, explanation_samples=20):

    try:

        save_dir = os.path.join(current_dir, 'figures', model_name, 'shap_plots')
        os.makedirs(save_dir, exist_ok=True)


        if isinstance(model, Pipeline):
            explainer_model = model.named_steps['model']
            X_train_data = model[:-1].transform(X_train)
            X_test_data = model[:-1].transform(X_test)
        else:
            explainer_model = model
            X_train_data = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
            X_test_data = X_test.values if isinstance(X_test, pd.DataFrame) else X_test


        print("⏳ Create a SHAP interpreter ")
        if isinstance(explainer_model, (RandomForestRegressor, XGBRegressor, LGBMRegressor)):
            explainer = shap.TreeExplainer(explainer_model)
        else:
            explainer = shap.KernelExplainer(explainer_model.predict, X_train_data)

        shap_values = explainer.shap_values(X_test_data)
        print("✅ The calculation of the SHAP values has been completed.")
        print("⏳ Draw the Shap summary chart")

        plt.figure()
        shap.summary_plot(shap_values, X_test_data, feature_names=feature_names, max_display=n_features, show=False)
        ax = plt.gca()
        ax.tick_params(axis='x', labelsize=26)
        ax.tick_params(axis='y', labelsize=26)
        x_label = ax.get_xlabel()
        if x_label:
            ax.set_xlabel(x_label, fontsize=26)
        plt.gcf().axes[-1].set_aspect(100)
        plt.title(f"SHAP Summary Plot - {model_name}", fontsize=28)
        plt.savefig(os.path.join(save_dir, f"{model_name}_shap_summary.tiff"),
                    dpi=600, format='tiff', bbox_inches='tight')
        plt.close()
        print("✅ Shap summary graph completed")



        print("⏳ Draw a feature clustering heatmap")
        try:
            plt.rcParams['font.family'] = 'serif'
            plt.rcParams['font.family'] = 'Times New Roman'
            plt.rcParams['axes.labelsize'] = 26
            plt.rcParams['axes.labelweight'] = 'bold'

            if len(shap_values.shape) == 2:
                global_shap_importance = np.mean(np.abs(shap_values), axis=0)
            else:
                global_shap_importance = np.mean(np.abs(shap_values), axis=0)

            sorted_indices = np.argsort(-global_shap_importance)[:n_features]
            selected_features = [feature_names[i] for i in sorted_indices]

            if len(shap_values.shape) == 2:
                shap_selected = shap_values[:, sorted_indices]
            else:
                shap_selected = shap_values[:, sorted_indices]

            corr_matrix = np.corrcoef(shap_selected.T)
            cmap = sns.diverging_palette(240, 1, as_cmap=True)

            plt.figure(figsize=(10, 8))
            grid = sns.clustermap(
                pd.DataFrame(corr_matrix,
                             index=selected_features,
                             columns=selected_features),
                cmap=cmap,
                annot=False,
                center=0,
                vmin=-1,
                vmax=1,
                linewidths=0.5,
                figsize=(10, 8),
                row_cluster=False,
                dendrogram_ratio=(0.15, 0.15)
            )
            n = len(selected_features)
            grid.ax_heatmap.set_yticks(np.arange(n) + 0.5)
            grid.ax_heatmap.set_yticklabels(selected_features, rotation=0, fontsize=26)
            grid.ax_heatmap.tick_params(axis='y', length=0)

            heatmap_pos = [0.15, 0.2, 0.7, 0.65]
            dendro_pos = [0.15, 0.85, 0.7, 0.1]
            cbar_pos = [0.05, 0.2, 0.02, 0.65]

            grid.ax_heatmap.set_position(heatmap_pos)
            grid.ax_col_dendrogram.set_position(dendro_pos)
            grid.cax.set_position(cbar_pos)

            for line in grid.ax_col_dendrogram.collections:
                line.set_linewidth(1.5)
            grid.ax_heatmap.set_xlabel("Features", weight='bold', fontsize=26)
            grid.ax_heatmap.set_ylabel("Features", weight='bold', fontsize=26)
            grid.ax_heatmap.set_xticklabels(
                grid.ax_heatmap.get_xticklabels(),
                rotation=45,
                ha='right',
                fontsize=26,
                weight='normal'
            )
            grid.ax_heatmap.set_yticklabels(
                grid.ax_heatmap.get_yticklabels(),
                fontsize=26,
                weight='normal'
            )

            grid.cax.yaxis.label.set_size(26)
            grid.cax.yaxis.label.set_weight('bold')
            grid.cax.yaxis.set_label_position('left')
            grid.cax.yaxis.label.set_rotation(270)
            grid.cax.yaxis.label.set_verticalalignment('bottom')
            grid.cax.yaxis.label.set_horizontalalignment('left')

            plt.title(f"Feature Clustering Heatmap - {model_name}\n(Based on SHAP Value Correlations)",
                      y=1.15,
                      fontsize=28,
                      fontweight='bold',
                      color='black')

            plt.setp(grid.ax_heatmap.get_xticklabels(), rotation=45, ha='right')

            grid.savefig(os.path.join(save_dir, f"{model_name}_feature_clustering.tiff"),
                         dpi=600,
                         format='tiff',
                         bbox_inches='tight')
            plt.close()
            print("✅ Feature clustering heatmap completed")
        except Exception as e:
            print(f"Feature clustering heatmap generation failed: {str(e)}")


        print("⏳ Draw the ordinary version of the feature clustering heatmap")
        try:
            plt.rcParams['font.family'] = 'Times New Roman'
            plt.rcParams['axes.labelsize'] = 16
            plt.rcParams['axes.labelweight'] = 'bold'

            global_shap_importance = np.mean(np.abs(shap_values), axis=0)
            sorted_indices = np.argsort(-global_shap_importance)[:n_features]
            selected_features = [feature_names[i] for i in sorted_indices]

            shap_selected = shap_values[:, sorted_indices]
            corr_matrix = np.corrcoef(shap_selected.T)
            corr_matrix_df = pd.DataFrame(corr_matrix,
                                          index=selected_features,
                                          columns=selected_features)
            cmap = sns.diverging_palette(240, 1, as_cmap=True)
            plt.figure(figsize=(10, 8))
            ax = sns.heatmap(
                corr_matrix_df,
                cmap=cmap,
                annot=False,
                center=0,
                vmin=-1,
                vmax=1,
                linewidths=0.5,
                square=True,
                cbar_kws={'shrink': 0.8, 'label': 'Correlation'}
            )

            ax.set_xlabel("Features", weight='bold', fontsize=26)
            ax.set_ylabel("Features", weight='bold', fontsize=26)

            ax.set_xticklabels(
                ax.get_xticklabels(),
                rotation=45,
                ha='right',
                fontsize=26,
                weight='normal'
            )
            ax.set_yticklabels(
                ax.get_yticklabels(),
                fontsize=26,
                weight='normal'
            )

            cbar = ax.collections.colorbar
            cbar.ax.tick_params(labelsize=26)
            cbar.set_label('Correlation', fontsize=26,
                           weight='bold', rotation=270, labelpad=25)

            plt.title(f"Feature Clustering Heatmap - {model_name}\n(Based on SHAP Value Correlations)",
                      y=1.15,
                      fontsize=28,
                      fontweight='bold',
                      color='black')

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{model_name}_feature_clustering.tiff"),
                        dpi=600,
                        format='tiff',
                        bbox_inches='tight')
            plt.close()
            print("✅ Feature clustering heatmap completed")
        except Exception as e:
            print(f"Feature clustering heatmap generation failed: {str(e)}")


        print("⏳ Draw the dependency graph of SHAP features")
        if len(shap_values.shape) == 2:

            plt.rcParams['font.family'] = 'Times New Roman'
            plt.rcParams['axes.labelweight'] = 'bold'

            for idx in sorted_indices:
                feature_name = feature_names[idx]
                plt.figure(figsize=(10, 8))

                shap.dependence_plot(
                    idx,
                    shap_values,
                    X_test_data,
                    feature_names=feature_names,
                    interaction_index='auto',
                    show=False,
                    dot_size=24
                )
                ax = plt.gca()
                ax.tick_params(axis='both', which='major', labelsize=26)
                ax.set_xlabel(ax.get_xlabel(), fontsize=26)
                ax.set_ylabel(ax.get_ylabel(), fontsize=26)
                cb = plt.gcf().axes[-1]
                cb.tick_params(labelsize=24)
                cb.set_ylabel(cb.get_ylabel(), fontsize=26)
                plt.title(
                    f"Feature Dependency - {feature_name}",
                    fontsize=28,
                    pad=10
                )

                plt.tight_layout()
                plt.savefig(
                    os.path.join(save_dir, f"{model_name}_dependence_{feature_name}.tiff"),
                    dpi=600,
                    format='tiff',
                    bbox_inches='tight'
                )
                plt.close()
            plt.rcParams.update(plt.rcParamsDefault)

        print("✅ SHAP feature dependency graph completion")


        print("⏳ Draw the SHAP clustering graph")

        plt.figure(figsize=(16, 12.8))
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['axes.labelweight'] = 'bold'


        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        feature_order = np.argsort(-mean_abs_shap)

        shap.plots.heatmap(
            shap.Explanation(
                values=shap_values,
                data=X_test_data,
                feature_names=feature_names
            ),
            max_display=n_features,
            feature_order=feature_order,
            show=False
        )
        ax = plt.gca()
        plt.title(f"SHAP Clustered Heatmap - {model_name}", fontsize=28,fontweight='bold',pad=10)
        ax.xaxis.label.set_size(26)
        ax.yaxis.label.set_size(26)
        ax.tick_params(axis='x', labelsize=26)

        cb = ax.figure.axes[-1]
        cb.yaxis.label.set_size(26)
        cb.yaxis.label.set_weight('bold')
        plt.savefig(os.path.join(save_dir, f"{model_name}_shap_heatmap2.tiff"),
                    dpi=600, format='tiff', bbox_inches='tight')
        plt.close()
        print("✅ SHAP clustering graph completed")
        print("⏳ Draw the SHAP interaction plot")

        if isinstance(explainer_model, (RandomForestRegressor, XGBRegressor)):
            try:
                interaction_values = shap.TreeExplainer(explainer_model).shap_interaction_values(X_test_data)
                if interaction_values is not None:
                    plt.figure()
                    shap.summary_plot(interaction_values, X_test_data,
                                      feature_names=feature_names,
                                      max_display=n_features,
                                      plot_type="compact_dot")
                    plt.title(f"Interaction Summary - {model_name}", fontsize=28)
                    plt.savefig(os.path.join(save_dir, f"{model_name}_shap_interaction.tiff"),
                                dpi=600, format='tiff', bbox_inches='tight')
                    plt.close()
            except Exception as e:
                print(f"Interaction graph generation failed:{str(e)}")
        print("✅ The SHAP interaction plot has been completed.")
        print(f"The SHAP analysis results have been saved as: {save_dir}")
        print(f"⏳ Draw a decision-making path diagram")
        print("⏳ Draw a decision-making path diagram")
        plt.figure(figsize=(10, 8))

        max_samples = min(sample_num, X_test_data.shape[0])
        sample_indices = np.random.choice(
            X_test_data.shape[0],
            size=max_samples,
            replace=False)
        if len(shap_values.shape) == 2:
            global_shap_importance = np.mean(np.abs(shap_values), axis=0)
        else:
            global_shap_importance = np.mean(np.abs(shap_values[0]), axis=0)

        sorted_feature_indices = np.argsort(-global_shap_importance)[:n_features]
        sorted_feature_names = [feature_names[i] for i in sorted_feature_indices]

        truncated_shap = shap_values[sample_indices][:, sorted_feature_indices]
        truncated_features = X_test_data[sample_indices][:, sorted_feature_indices]

        shap.decision_plot(
            explainer.expected_value,
            truncated_shap,
            truncated_features,
            feature_names=sorted_feature_names,
            show=False )
        current_axis = plt.gca()
        current_axis.set_xlim(left=None, right=100)
        ax = plt.gca()

        for spine in ax.spines.values():

            spine.set_linewidth(3)
        for line in ax.lines:
            if line.get_linestyle() == '-':
                line.set_linewidth(2)
        ax.tick_params(axis='x',which='major',labelsize=26)
        ax.tick_params(axis='y',which='both', labelsize=26)
        plt.draw()
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(26 if label.get_position() != 0 else 26)
            label.set_fontname('Times New Roman')
        ax.xaxis.label.set_fontsize(26)
        ax.xaxis.label.set_fontname('Times New Roman')
        ax.xaxis.label.set_weight('bold')
        ax.yaxis.label.set_fontname('Times New Roman')
        plt.savefig(
            os.path.join(save_dir, f"{model_name}_decision_plot_top.tiff"),dpi=600,format='tiff',bbox_inches='tight')
        plt.close()
        print(f"✅ The merged decision path diagram is completed")


        print(f"⏳ Draw the force diagrams for the first{explanation_samples}samples")
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 26
        for i in range(min(explanation_samples, len(X_test_data))):
            plt.figure(figsize=(10, 3))
            shap.force_plot(
                explainer.expected_value,
                shap_values[i],
                X_test_data[i],
                feature_names=feature_names,
                matplotlib=True,
                show=False
            )
            ax = plt.gca()

            for text in ax.texts:
                if '=' in text.get_text():
                    feature_name = text.get_text().split('=')[0]
                    text.set_text(feature_name)
                    text.set_fontsize(24)

            plt.title(
                f"Force Plot - Sample {i} - {model_name}",
                fontsize=26,
                fontweight='bold',
                pad=30
            )


            ax.tick_params(axis='x', labelsize=26)
            plt.title(f"Force Plot - Sample {i} - {model_name}", fontsize=24)
            plt.savefig(
                os.path.join(save_dir, f"{model_name}_force_plot_sample_{i}.tiff"),
                dpi=600, format='tiff', bbox_inches='tight'
            )
            plt.close()
        print("✅ The attempt to complete by a single sample")
    except Exception as e:
        print(f"An error occurred during the SHAP analysis: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    set_seed(42)
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)

    best_parameters = {
        'SVR': {'C': 80, 'coef0': 1.0, 'epsilon': 0.06308564123331233, 'degree': 5, 'gamma': 0.1, 'kernel': 'rbf'},
        'RF': {'n_estimators': 883, 'max_depth': 10, 'max_features': 0.6, 'min_samples_split': 3,
               'min_samples_leaf': 2},

        'XGBoost': {'n_estimators': 500, 'max_depth': 10, 'learning_rate': 0.134,
                    'subsample': 1.0, 'colsample_bytree': 1.0, 'reg_alpha': 0.0,
                    'reg_lambda': 1.4, 'gamma': 0.0},
        'LightGBM': {'feature_fraction': 0.8138257486166964,'learning_rate': 0.09568447787904698,'max_depth': 7,'min_child_samples': 6,
                     'n_estimators': 1500,'num_leaves': 49,'reg_alpha': 0.1526852115104169,'reg_lambda': 0.7182500013327633,'subsample': 0.6015050137547598,'verbosity':-1},
        'KNN': {'n_neighbors': 5, 'weights': 'distance', 'algorithm': 'auto'},
        'Lasso': {'alpha': 0.001, 'fit_intercept': True},
        'LR': {'fit_intercept': True},
        'GPR': {'kernel': 'RBF', 'alpha': 0.01, 'n_restarts_optimizer': 10}
    }

    train_path = os.path.join(current_dir, "datas", "training_data.xlsx")
    _, features_all, output_all = read_excel_data(train_path, use_header=True)
    X_train, y_train, X_test, y_test = preprocess_data(features_all, output_all)

    while True:
        try:
            opt_choice = int(input("\nPlease select the optimization mode:\n1. Carry out Bayesian optimization\n2. Use the optimized parameters\nPlease enter a number (1/2): "))
            if opt_choice in [1, 2]:
                break
            else:
                print("Please enter 1 or 2!")
        except ValueError:
            print("Invalid input. Please enter again!")

    model_options = {
        1: 'SVR', 2: 'RF', 3: 'XGBoost', 4: 'LightGBM',
        5: 'KNN', 6: 'Lasso', 7: 'LR'
    }

    while True:
        try:
            print("\nList of available models:")
            for num, name in model_options.items():
                print(f"{num}. {name}")
            model_choice = int(input("Please enter the model number (1-7): "))
            if model_choice in model_options:
                model_name = model_options[model_choice]
                break
            else:
                print("Please enter a number between 1 and 7!")
        except ValueError:
            print("Invalid input. Please enter again!")

    param_spaces = {
        'SVR': {
            'model__C': Real(0.1, 80, prior='log-uniform'),
            'model__kernel': Categorical(['rbf', 'linear', 'poly']),
            'model__gamma': Real(1e-4, 0.1, prior='log-uniform'),
            'model__degree': Integer(2, 5),
            'model__coef0': Real(-1.0, 1.0),
            'model__epsilon': Real(0.01, 0.2)
        },
        'RF': {
            'model__n_estimators': Integer(200, 1000),
            'model__max_depth': Integer(5, 10),
            'model__max_features': Categorical(["sqrt", 0.3, 0.6, 0.9]),
            'model__min_samples_split': Integer(2, 50),
            'model__min_samples_leaf': Integer(2, 10),

        },
        'XGBoost': {
            'model__n_estimators': Integer(50, 500),
            'model__max_depth': Integer(3, 8),
            'model__learning_rate': Real(0.01, 0.2),
            'model__subsample': Real(0.3, 1.0),
            'model__colsample_bytree': Real(0.3, 1.0),
            'model__reg_alpha': Real(0, 5),
            'model__reg_lambda': Real(0, 5),
            'model__gamma': Real(0, 3),
            'model__min_child_weight': Real(1e-5, 10)
        },
        'LightGBM': {
            'model__n_estimators': Integer(100, 1500),
            'model__num_leaves': Integer(10, 400),
            'model__max_depth': Integer(3, 10),
            'model__learning_rate': Real(0.001, 0.2),
            'model__subsample': Real(0.6, 1.0),
            'model__reg_alpha': Real(0, 1),
            'model__reg_lambda': Real(0, 1),
            'model__feature_fraction': Real(0.6, 1.0),
            'model__min_child_samples': Integer(5, 100),
            'model__verbosity': Categorical([-1])
        },
        'KNN': {
            'model__n_neighbors': Integer(1, 50),
            'model__weights': Categorical(['uniform', 'distance']),
            'model__algorithm': Categorical(['auto', 'ball_tree', 'kd_tree', 'brute']),
            'model__leaf_size': Integer(10, 100)
        },
        'Lasso': {
            'model__alpha': Real(0.001, 1, prior='log-uniform'),
            'model__fit_intercept': Categorical([True, False]),
            'model__max_iter': Integer(500, 5000)
        },
        'LR': {
            'model__fit_intercept': Categorical([True, False])
        },
        'GPR': {
            'model__kernel': Categorical(['RBF', 'Matern', 'RationalQuadratic']),
            'model__alpha': Real(1e-5, 1e-1, prior='log-uniform'),
            'model__n_restarts_optimizer': Integer(5, 20)
        }
    }



    def create_gpr_kernel(kernel_name):
        from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, ConstantKernel as C
        kernel_map = {
            'RBF': C(1.0) * RBF(10.0),
            'Matern': C(1.0) * Matern(nu=1.5),
            'RationalQuadratic': C(1.0) * RationalQuadratic()
        }
        return kernel_map.get(kernel_name, RBF())


    if opt_choice == 1:
        print(f"\nThe Bayesian optimization is currently being executed ({model_name})...")
        if model_name not in param_spaces:
            raise ValueError(f"{model_name} Parameter space has not been configured yet!")

        model_initializers = {
            'SVR': Pipeline([('scaler', StandardScaler()), ('model', SVR())]),
            'RF': Pipeline([('scaler', StandardScaler()), ('model', RandomForestRegressor(random_state=42))]),
            'XGBoost': Pipeline([('scaler', StandardScaler()), ('model', XGBRegressor(random_state=42))]),
            'LightGBM': Pipeline([('scaler', StandardScaler()), ('model', LGBMRegressor(random_state=42))]),
            'KNN': Pipeline([('scaler', StandardScaler()), ('model', KNeighborsRegressor())]),
            'Lasso': Pipeline([('scaler', StandardScaler()), ('model', Lasso(random_state=42))]),
            'LR': Pipeline([('scaler', StandardScaler()), ('model', LinearRegression())]),
            'GPR': Pipeline([('scaler', StandardScaler()), ('model', GaussianProcessRegressor(random_state=42))])
        }

        if model_name == 'GPR':
            def gpr_model_params(**params):
                return Pipeline([
                    ('scaler', StandardScaler()),
                    ('model', GaussianProcessRegressor(
                        kernel=create_gpr_kernel(params['model__kernel']),
                        alpha=params['model__alpha'],
                        n_restarts_optimizer=params['model__n_restarts_optimizer'],
                        random_state=42
                    ))
                ])


            best_model = grid_search_model(Pipeline([('scaler', StandardScaler()), ('model', SVR())]),X_train,y_train)
        else:
            base_model = model_initializers[model_name]
            best_model = grid_search_model(base_model, param_spaces[model_name], X_train, y_train)
    else:
        print(f"\nLoading optimization parameters({model_name})...")
        try:

            raw_params = best_parameters[model_name]

            print("\n==============================")
            print(f"[Baseline] Train using the default initial parameters {model_name}")
            print("==============================")

            model_classes = {
                'SVR': SVR,
                'RF': RandomForestRegressor,
                'XGBoost': XGBRegressor,
                'LightGBM': LGBMRegressor,
                'KNN': KNeighborsRegressor,
                'Lasso': Lasso,
                'LR': LinearRegression,
                'GPR': GaussianProcessRegressor
            }

            valid_params = {}
            if model_name in model_classes:
                model_class = model_classes[model_name]
                valid_params = {k: v for k, v in raw_params.items() if k in model_class().get_params()}
                if model_name in ['RF', 'XGBoost', 'LightGBM', 'GPR']:
                    valid_params['random_state'] = 42

                if model_name == 'GPR':
                    valid_params['kernel'] = create_gpr_kernel(raw_params['kernel'])

                print(f"Effective parameters: {valid_params}")

                best_model = model_class(**valid_params)
            else:
                raise ValueError(f"Unknown model: {model_name}")

        except KeyError as e:
            print(f"Parameter error:，Use default parameters!")
            best_model = model_classes[model_name]()

    print(f"\nStart training the {model_name} model...")
    try:
        trained_model, metrics = train_and_evaluate_model(
            best_model, X_train, y_train, X_test, y_test, model_name, save_pred=True
        )
        print(f"\nCarry out stability analysis of cross-validation...")
        cross_val_stability(trained_model,np.vstack([X_train, X_test]),np.concatenate([y_train, y_test]),cv=5,model_name=model_name)
        plot_learning_curve(trained_model, X_train, y_train, model_name)

        predict_and_plot_lines(trained_model,  model_name, current_dir)
        reference_and_plot_lines(trained_model, model_name, current_dir)
        feature_labels = features_all.columns.tolist()
        plot_feature_importance(trained_model, X_train, y_train, feature_labels)

        if isinstance(trained_model, Pipeline):
            model_step = trained_model.named_steps['model']
        else:
            model_step = trained_model
        feature_labels = features_all.columns.tolist()
        shap_choice = input("Do you want to perform SHAP analysis? (1 = Yes, others = No):")
        if shap_choice == '1':
            perform_shap_analysis(trained_model, X_train, X_test,
                                  feature_labels, model_name, current_dir,
                                  n_features=15)




        print(f"\n{model_name} Model training process completed!")
    except Exception as e:
        print(f"Model training failed: {str(e)}")
        traceback.print_exc()
