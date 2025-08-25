import os
import uuid
from typing import List, Tuple, Optional, Dict
from sklearn.impute import KNNImputer
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore")

PLOT_DIR = os.path.join("static", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

EXPORT_DIR = os.path.join("static", "exports")
os.makedirs(EXPORT_DIR, exist_ok=True)


def _unique_png(prefix: str):
    fname = f"{prefix}_{uuid.uuid4().hex[:8]}.png"
    abs_path = os.path.join(PLOT_DIR, fname)  
    rel_path = f"plots/{fname}"        
    return abs_path, rel_path

def _unique_csv(prefix: str):
    fname = f"{prefix}_{uuid.uuid4().hex[:8]}.csv"
    abs_path = os.path.join(EXPORT_DIR, fname)
    rel_path = f"exports/{fname}"   
    return abs_path, rel_path


def read_any_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".csv", ".txt"]:
        return pd.read_csv(path)
    elif ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def knn_impute_df(
    df: pd.DataFrame,
    target_col: str,
    n_neighbors: int = 3,
    show_time_col: str | None = "time"
) -> dict:
    """
    Impute all numeric columns with KNN (n_neighbors).
    Return: filled DataFrame, list of replacements for target_col, and plots.
    """

    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    non_num_cols = [c for c in df.columns if c not in num_cols]

    df_orig = df.copy()

    imputer = KNNImputer(n_neighbors=n_neighbors)
    if len(num_cols) == 0:
        df_filled = df.copy()
    else:
        num_block = df[num_cols]
        num_filled = pd.DataFrame(imputer.fit_transform(num_block), columns=num_cols, index=df.index)
        df_filled = df.copy()
        df_filled[num_cols] = num_filled

    rep_rows = df[target_col].isna() if target_col in df.columns else pd.Series([], dtype=bool)
    replaced_idx = df.index[rep_rows].tolist()
    replacements = []
    for i in replaced_idx:
        excel_line = int(i) + 2  # header + 1-based
        time_val = df.loc[i, show_time_col] if (show_time_col and show_time_col in df.columns) else None
        orig_val = df.loc[i, target_col] if target_col in df.columns else None
        imp_val = df_filled.loc[i, target_col] if target_col in df_filled.columns else None
        replacements.append({
            "excel_line": excel_line,
            "index": int(i),
            "time": None if pd.isna(time_val) else time_val,
            "original": None if pd.isna(orig_val) else float(orig_val),
            "imputed": None if pd.isna(imp_val) else float(imp_val),
        })

    rel_series_plot = rel_dist_plot = rel_corr_before = rel_corr_after = None
    try:
        # 1) Series plot: target original vs imputed
        if target_col in df_filled.columns and target_col in df_orig.columns:
            y_orig = pd.to_numeric(df_orig[target_col], errors="coerce")
            y_imp  = pd.to_numeric(df_filled[target_col], errors="coerce")
            x = np.arange(len(df_filled))

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(x, y_imp, marker='o', label=f"{target_col} (KNN imputed)")
            # Mark missing points that were imputed
            if len(replaced_idx) > 0:
                ax.scatter(replaced_idx, y_imp.iloc[replaced_idx], color="red", zorder=5, label="Imputed points")
            ax.set_title(f"{target_col} over index (KNN Imputed, k={n_neighbors})")
            ax.set_xlabel("Index")
            ax.set_ylabel(target_col)
            ax.grid(True, alpha=0.3)
            ax.legend()
            abs_p, rel_series_plot = _unique_png("knn_series")
            plt.tight_layout(); fig.savefig(abs_p, dpi=140); plt.close(fig)

        # 2) Distribution comparison (original non-NaN vs imputed-at-missing)
        if target_col in df_filled.columns and target_col in df_orig.columns:
            orig_non_nan = pd.to_numeric(df_orig[target_col], errors="coerce").dropna().values
            imp_only = df_filled.loc[replaced_idx, target_col].values if len(replaced_idx) else np.array([])

            fig2, ax2 = plt.subplots(figsize=(8, 5))
            if orig_non_nan.size > 0:
                ax2.hist(orig_non_nan, bins=20, density=True, alpha=0.5, label="Original (non-NaN)")
            if imp_only.size > 0:
                ax2.hist(imp_only, bins=20, density=True, alpha=0.5, label="Imputed (at missing)")
            ax2.set_title(f"Distribution: Original vs Imputed ({target_col})")
            ax2.set_xlabel(target_col); ax2.set_ylabel("Density")
            ax2.legend()
            abs_p2, rel_dist_plot = _unique_png("knn_dist")
            plt.tight_layout(); fig2.savefig(abs_p2, dpi=140); plt.close(fig2)

        # 3) Correlation BEFORE
        if len(num_cols) > 0:
            corr_before = df_orig[num_cols].corr()
            fig3, ax3 = plt.subplots(figsize=(10, 5))
            cax = ax3.imshow(corr_before, vmin=-1, vmax=1, cmap="coolwarm")
            ax3.set_title("Correlation (Before Imputation)")
            ax3.set_xticks(range(len(num_cols))); ax3.set_yticks(range(len(num_cols)))
            ax3.set_xticklabels(num_cols, rotation=90)
            ax3.set_yticklabels(num_cols)
            fig3.colorbar(cax, ax=ax3, fraction=0.046, pad=0.04)
            abs_p3, rel_corr_before = _unique_png("corr_before")
            plt.tight_layout(); fig3.savefig(abs_p3, dpi=140); plt.close(fig3)

        # 4) Correlation AFTER
        if len(num_cols) > 0:
            corr_after = df_filled[num_cols].corr()
            fig4, ax4 = plt.subplots(figsize=(10, 5))
            cax2 = ax4.imshow(corr_after, vmin=-1, vmax=1, cmap="coolwarm")
            ax4.set_title("Correlation (After KNN Imputation)")
            ax4.set_xticks(range(len(num_cols))); ax4.set_yticks(range(len(num_cols)))
            ax4.set_xticklabels(num_cols, rotation=90)
            ax4.set_yticklabels(num_cols)
            fig4.colorbar(cax2, ax=ax4, fraction=0.046, pad=0.04)
            abs_p4, rel_corr_after = _unique_png("corr_after")
            plt.tight_layout(); fig4.savefig(abs_p4, dpi=140); plt.close(fig4)

    except Exception:
        pass

    # Save a downloadable filled CSV
    abs_csv, rel_csv = _unique_csv("filled_knn")
    try:
        df_filled.to_csv(abs_csv, index=False)
    except Exception:
        rel_csv = None

    return {
        "df_filled": df_filled,
        "replacements": replacements,         # list of dicts
        "plots": {
            "series": rel_series_plot,
            "distribution": rel_dist_plot,
            "corr_before": rel_corr_before,
            "corr_after": rel_corr_after
        },
        "download_csv": rel_csv,
        "n_neighbors": n_neighbors,
        "num_cols": num_cols
    }


def detect_anomalies(series: pd.Series, threshold: float = 0.20, rows_to_check: int | None = None) -> dict:
    s = pd.to_numeric(series, errors="coerce")
    if rows_to_check is not None:
        s = s.head(rows_to_check)
    df = pd.DataFrame({"value": s}).dropna().reset_index(drop=True)

    n = len(df)
    if n < 2:
        return {"count": 0, "threshold": threshold, "anomalies": [], "plot_series": None, "plot_hist": None}

    anomalies = []
    i = 1
    while i < n:
        prev = df.iloc[i - 1, 0]
        curr = df.iloc[i, 0]
        if pd.notna(prev) and pd.notna(curr) and prev != 0:
            change = (curr - prev) / prev
            if abs(change) > threshold:
                anomalies.append((int(i + 1), int(i), float(curr), float(round(change, 3))))
                i += 2
                continue
        i += 1

    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(range(n), df["value"].values, marker="o", label="Series")
        for _, idx, val, _ in anomalies:
            ax.scatter(idx, val, color="red", zorder=5)
        ax.set_title("Series with Detected Anomalies")
        ax.set_xlabel("Index"); ax.set_ylabel("Value"); ax.grid(True, alpha=0.3); ax.legend()
        abs_series, rel_series = _unique_png("anoms")
        plt.tight_layout(); fig.savefig(abs_series, dpi=140); plt.close(fig)

        changes = []
        for j in range(1, n):
            a = df.iloc[j - 1, 0]; b = df.iloc[j, 0]
            if pd.notna(a) and pd.notna(b) and a != 0:
                changes.append((b - a) / a)
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        ax2.hist(changes, bins=30, edgecolor="black")
        ax2.axvline(threshold, ls="--", color="red"); ax2.axvline(-threshold, ls="--", color="red")
        ax2.set_title("Distribution of % Changes"); ax2.set_xlabel("Percentage change"); ax2.set_ylabel("Frequency")
        abs_hist, rel_hist = _unique_png("change_hist")
        plt.tight_layout(); fig2.savefig(abs_hist, dpi=140); plt.close(fig2)

    except Exception:
        rel_series = rel_hist = None

    return {
        "count": len(anomalies),
        "threshold": threshold,
        "anomalies": anomalies,
        "plot_series": rel_series,  
        "plot_hist": rel_hist       
    }


def linear_regression_next(
    series: pd.Series,
    window_size: int = 3,
    rows: Optional[int] = None
) -> Dict:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if rows is not None:
        s = s.head(rows)
    y = s.values
    if len(y) <= window_size + 1:
        raise ValueError("Not enough data for the chosen window size.")

    X, target = [], []
    for i in range(len(y) - window_size):
        X.append(y[i:i+window_size])
        target.append(y[i+window_size])
    X = np.array(X)
    target = np.array(target)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = LinearRegression()
    model.fit(Xs, target)

    last_vals = y[-window_size:].reshape(1, -1)
    pred = float(model.predict(scaler.transform(last_vals))[0])

    pred_train = model.predict(Xs)
    
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(target, label="Actual y", color="navy")
    ax.plot(pred_train, label="Predicted y", color="green")
    ax.set_title("Model Fit on Training Data (Linear Regression)")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3)
    ax.legend()
    abs_fit, rel_fit = _unique_png("lin_fit")
    plt.tight_layout()
    fig.savefig(abs_fit, dpi=140)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(6, 5))
    ax2.plot(np.arange(len(y)), y, marker="o", label="Known Series")
    ax2.scatter(len(y), pred, color="red", zorder=5, label="Predicted Next")
    ax2.axvline(len(y), ls="--", color="red", alpha=0.6)
    ax2.set_title("Next-Value Forecast (Linear Regression)")
    ax2.set_xlabel("Index")
    ax2.set_ylabel("Value")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    abs_next, rel_next = _unique_png("lin_next")
    plt.tight_layout()
    fig2.savefig(abs_next, dpi=140)
    plt.close(fig2)

    return {
        "window_size": window_size,
        "n_train": len(target),
        "prediction": float(pred),
        "plot_fit": rel_fit,    
        "plot_next": rel_next,   
    }



def sarimax_next(
    series: pd.Series,
    order=(1,1,1),
    seasonal_order=(0,0,0,0),
    rows: Optional[int] = None
) -> Dict:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if rows is not None:
        s = s.head(rows)
        
    y = s.values
    if len(y) < 10:
        raise ValueError("At least 10 points recommended for SARIMAX.")

    model = SARIMAX(y, order=order, seasonal_order=seasonal_order)
    res = model.fit(disp=False)
    forecast = float(res.forecast(steps=1)[0])

    fig, ax = plt.subplots(figsize=(6,5))
    ax.plot(np.arange(len(y)), y, marker="o", label="Known Series")
    ax.scatter(len(y), forecast, color="crimson", zorder=5, label="SARIMAX Next")
    ax.axvline(len(y), ls="--", color="crimson", alpha=0.6)
    ax.set_title(f"Next-Value Forecast (SARIMAX {order}, seasonal {seasonal_order})")
    ax.set_xlabel("Index"); ax.set_ylabel("Value"); ax.grid(True, alpha=0.3); ax.legend()
    abs_next, rel_next = _unique_png("sarimax_next")
    plt.tight_layout(); fig.savefig(abs_next, dpi=140); plt.close(fig)

    return {
        "order": order,
        "seasonal_order": seasonal_order,
        "prediction": forecast,
        "plot_next": rel_next,     
        "aic": float(getattr(res, "aic", np.nan)),
        "bic": float(getattr(res, "bic", np.nan)),
    }
