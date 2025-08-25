import os
import time
import json
import math
import random
import threading
from collections import deque

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, Response
from markupsafe import Markup
from werkzeug.utils import secure_filename
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from analysis import (
    read_any_table,          
    detect_anomalies,         
    linear_regression_next,   
    sarimax_next,            
    knn_impute_df             
)

ALLOWED = {".csv", ".txt", ".xlsx", ".xls"}
UPLOAD_DIR = "uploads"

app = Flask(__name__)
app.secret_key = "change-this"  # set via env in production
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024
os.makedirs(UPLOAD_DIR, exist_ok=True)

def allowed_file(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in ALLOWED


LIVE_MAX = 300        
_live_buf = deque(maxlen=LIVE_MAX)  
_live_t = 0
_live_lock = threading.Lock()


FAULT_EVERY = 20        
INJECT_NAN_EVERY = 120      
BIG_JUMP_UP  = (1.8, 2.4)   
BIG_JUMP_DOWN= (0.30, 0.50) 
MIN_FLOOR    = 0.05         


def _synth_value(t: int) -> float:
    """Daily + weekly pattern + noise + occasional spikes."""
    daily  = 2.8 + 0.6 * math.sin(2 * math.pi * (t % 86400) / 86400.0)        # 1-day cycle
    weekly = 0.25 * math.sin(2 * math.pi * (t % (86400 * 7)) / (86400.0 * 7)) # 7-day cycle
    noise  = random.gauss(0, 0.04)
    val    = daily + weekly + noise
    if random.random() < 0.02:  # rare event
        val += random.choice([0.4, 0.7, -0.3])
    return max(0.1, round(val, 4))

def _inject_fault(value: float, step_index: int) -> tuple[float, str | None]:
    """
    Occasionally corrupt the value to trigger anomalies and missing-data paths.
    Returns (possibly_faulted_value, fault_mode or None).
    """
    if step_index > 0 and step_index % INJECT_NAN_EVERY == 0:
        return float("nan"), "nan"

    if step_index > 0 and step_index % FAULT_EVERY == 0:
        mode = random.choice(["jump_up", "jump_down", "drop_zero"])
        if mode == "jump_up":
            return value * random.uniform(*BIG_JUMP_UP), mode
        if mode == "jump_down":
            return max(MIN_FLOOR, value * random.uniform(*BIG_JUMP_DOWN)), mode
        if mode == "drop_zero":
            return MIN_FLOOR, mode

    return value, None

def _predict_next_lr(series: list[float], window=3, train_len=60) -> float | None:
    """Sliding-window Linear Regression (fast, retrained each tick)."""
    if len(series) < window + 2:
        return None
    y = np.array(series[-train_len:], dtype=float)
    if len(y) <= window:
        return None
    X, target = [], []
    for i in range(len(y) - window):
        X.append(y[i:i+window])
        target.append(y[i+window])
    X = np.array(X); target = np.array(target)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = LinearRegression().fit(Xs, target)
    last = np.array(series[-window:], dtype=float).reshape(1, -1)
    return float(model.predict(scaler.transform(last))[0])


def fill_missing_values_live(series: pd.Series):
    """
    Reuse your KNN imputer on a one-column DataFrame.
    Returns (filled_series, imputation_info_dict).
    """
    df_tmp = pd.DataFrame({"value": pd.to_numeric(series, errors="coerce")})
    # show_time_col=None because live series has no explicit time column
    imp = knn_impute_df(df_tmp, target_col="value", n_neighbors=3, show_time_col=None)
    filled = imp["df_filled"]["value"]
    return filled, imp


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        f = request.files.get("file")
        if not f or f.filename == "":
            flash("Please choose a file.")
            return redirect(url_for("index"))
        if not allowed_file(f.filename):
            flash("Unsupported file type.")
            return redirect(url_for("index"))

        filename = secure_filename(f.filename)
        path = os.path.join(UPLOAD_DIR, filename)
        f.save(path)

        try:
            df = read_any_table(path)
        except Exception as e:
            flash(f"Failed to read file: {e}")
            return redirect(url_for("index"))

        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        other_cols = df.columns.difference(num_cols).tolist()

        if num_cols:
            preview = knn_impute_df(df, target_col=num_cols[0], n_neighbors=3, show_time_col="time")
            df_filled = preview["df_filled"]
        else:
            df_filled = df.copy()

        orig_html   = Markup(df.to_html(classes="table table-striped table-sm", index=False))
        filled_html = Markup(df_filled.to_html(classes="table table-striped table-sm", index=False))

        return render_template(
            "index.html",
            uploaded=True,
            filename=filename,
            numeric_columns=num_cols,
            other_columns=other_cols,
            orig_table=orig_html,
            filled_table=filled_html
        )

    return render_template("index.html", uploaded=False)


@app.route("/analyze", methods=["POST"])
def analyze():
    filename = request.form.get("filename")
    column   = request.form.get("column")
    threshold= float(request.form.get("threshold", 0.20))
    win      = int(request.form.get("window_size", 3))
    ar       = int(request.form.get("ar", 1))
    diff     = int(request.form.get("diff", 1))
    ma       = int(request.form.get("ma", 1))
    seas_p   = request.form.get("seasonal", "false") == "true"

    path = os.path.join(UPLOAD_DIR, filename)
    df = read_any_table(path)

    if column not in df.columns:
        flash("Selected column not found.")
        return redirect(url_for("index"))

    # --- KNN impute the whole DF and use the filled target column downstream
    imp = knn_impute_df(df, target_col=column, n_neighbors=3, show_time_col="time")
    series_imputed = imp["df_filled"][column]

    # 1) anomalies (on imputed series)
    anom = detect_anomalies(series_imputed, threshold=threshold, rows_to_check=None)

    # 2) linear regression (on imputed)
    lin = linear_regression_next(series_imputed, window_size=win, rows=None)

    # 3) sarimax (on imputed)
    seasonal_order = (0, 0, 0, 0)
    if seas_p:
        seasonal_order = (1, 0, 1, 12)
    sar = sarimax_next(series_imputed, order=(ar, diff, ma), seasonal_order=seasonal_order, rows=None)

    # rows actually analyzed
    nrows = len(pd.to_numeric(series_imputed, errors="coerce").dropna())


    next_idx_zero_based = nrows          # where the next prediction would be placed (0-based)
    next_row_one_based  = nrows + 1      # 1-based row number in the DataFrame
    excel_line_next     = nrows + 2      # Excel-style line (1 header + 1 for zero index)
    
    
    return render_template(
        "results.html",
        filename=filename,
        column=column,
        nrows=nrows,
        impute=imp,
        anom=anom,
        lin=lin,
        sar=sar,
        next_row_one_based=next_row_one_based,
        excel_line_next=excel_line_next
    )


# -------------------------------
# Live: full pipeline on the stream
# -------------------------------
def analyze_live_data():
    with _live_lock:
        ser = pd.Series(list(_live_buf), name="value")

    # 1) KNN imputation on the live series
    ser_filled, imp = fill_missing_values_live(ser)

    # 2) Anomaly detection
    anom = detect_anomalies(ser_filled, threshold=0.20, rows_to_check=None)

    # 3) Predictions
    lin  = linear_regression_next(ser_filled, window_size=3, rows=None)
    sar  = sarimax_next(ser_filled, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0), rows=None)

    return {"impute": imp, "anom": anom, "lin": lin, "sar": sar}


@app.route("/live/stream")
def live_stream():
    def event_stream():
        global _live_t

        with _live_lock:
            if not _live_buf:
                for _ in range(60):
                    _live_buf.append(_synth_value(_live_t))
                    _live_t += 5

        while True:
            time.sleep(2)

            with _live_lock:
                v = _synth_value(_live_t)
                count_before = len(_live_buf)
                v, fault_mode = _inject_fault(v, count_before + 1)
                _live_buf.append(v)
                current_t = _live_t
                count = len(_live_buf)
                _live_t += 2

            res = analyze_live_data()
            anom_idx = [a[1] for a in res["anom"]["anomalies"]][-50:]  
            
            imp_list = res["impute"]["replacements"] or []
            imputed_idx = [r.get("index") for r in imp_list if r.get("index") is not None][-50:]

            payload = {
                "t": current_t,
                "count": count,
                "value": v,
                "pred_lr":  res["lin"]["prediction"],
                "pred_sar": res["sar"]["prediction"],
                "anom_count": res["anom"]["count"],
                "anom_idx": anom_idx,
                "imputed_idx": imputed_idx,
                "imputed_total": len(res["impute"]["replacements"]),
                "threshold": res["anom"]["threshold"],
                "fault_mode": fault_mode 
            }
            yield f"data: {json.dumps(payload)}\n\n"

    return Response(
        event_stream(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


@app.route("/live")
def live_page():
    return render_template("live.html")


if __name__ == "__main__":
    app.run(debug=True, port=5000)
