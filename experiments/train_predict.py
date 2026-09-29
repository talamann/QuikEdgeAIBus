import argparse
import datetime
import json
import os
import time
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import RMSE
from neuralforecast.models import PatchTST, DLinear as NativeDLinear
from sklearn.metrics import r2_score

from dlinear_model import DLinear, build_windows

BITBRAINS_PATH = os.path.join("datasets", "bitbrains", "rnd")
MONTHS = ["2013-7", "2013-8", "2013-9"]
FILES = ["383.csv", "392.csv", "386.csv"]
UNIQUE_CORES = 3
SPLIT_INDEX = 70731


def df_processing(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.replace("\t", "")
    df["DateTime"] = df["Timestamp [ms]"].apply(
        lambda x: datetime.datetime.fromtimestamp(x).replace(second=0, microsecond=0)
    )
    df.set_index("DateTime", inplace=True)
    df = df.drop(columns=["Timestamp [ms]"]).resample("5min").ffill()
    return df


def fill_missing(df: pd.DataFrame, start, end) -> pd.DataFrame:
    previous_day_data = df[
        (df.index >= start - pd.DateOffset(days=1))
        & (df.index <= end - pd.DateOffset(days=1))
    ]
    missing_period_timestamps = pd.date_range(start=start, end=end, freq="5min")
    replicated_data = previous_day_data.copy()
    replicated_data.index = missing_period_timestamps[: len(previous_day_data)]
    df_filled = pd.concat([df, replicated_data]).sort_index()
    return df_filled


def load_bitbrains(path) -> pd.DataFrame:
    dfs = {file: [] for file in FILES}
    for month in MONTHS:
        for file in FILES:
            file_path = os.path.join(path, month, file)
            df = pd.read_csv(file_path, sep=";")
            dfs[file].append(df_processing(df))
    dfs = {file: pd.concat(dfs[file]).bfill() for file in FILES}

    start_dates = {
        "2013-07-30 23:00:00": pd.to_datetime("2013-07-31 23:00:00"),
        "2013-08-30 23:00:00": pd.to_datetime("2013-08-31 23:00:00"),
    }
    for start, end in start_dates.items():
        for key in dfs:
            dfs[key] = fill_missing(dfs[key], pd.to_datetime(start), end)

    merged_df = pd.concat(dfs.values()).sort_index()
    df = merged_df[["CPU cores", "CPU usage [%]"]]
    df = df[df.index > "2013-06-30 23:55:00"]
    df.reset_index(inplace=True)
    df.sort_values(by=["index", "CPU cores"], inplace=True)
    df.rename(
        columns={"CPU cores": "unique_id", "index": "ds", "CPU usage [%]": "y"},
        inplace=True,
    )
    return df


def split_dataset(df: pd.DataFrame):
    if df.shape[0] == 78588:
        split = SPLIT_INDEX
    else:
        split = (int(df.shape[0] * 0.9) // UNIQUE_CORES) * UNIQUE_CORES
    assert (df.shape[0] - split) % UNIQUE_CORES == 0
    df_train = df[:split].reset_index(drop=True)
    df_test = df[split:].reset_index(drop=True)
    print(f"[data] total={df.shape[0]} train={df_train.shape[0]} test={df_test.shape[0]}")
    return df_train, df_test


def core_arrays(df: pd.DataFrame) -> Dict[int, np.ndarray]:
    out = {}
    for uid in sorted(df["unique_id"].unique()):
        out[int(uid)] = df.loc[df["unique_id"] == uid, "y"].to_numpy(dtype=np.float32)
    return out


def train_dlinear(df_train: pd.DataFrame, args, device) -> DLinear:
    arrays = core_arrays(df_train)
    X_chunks, Y_chunks = [], []
    for uid in sorted(arrays):
        X, Y = build_windows(arrays[uid], args.input_size, args.h)
        X_chunks.append(X)
        Y_chunks.append(Y)
    X = np.concatenate(X_chunks)
    Y = np.concatenate(Y_chunks)

    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = DLinear(args.input_size, args.h, args.moving_avg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    model.train()
    for epoch in range(1, args.epochs + 1):
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * xb.shape[0]
        if args.verbose:
            print(f"[dlinear] epoch {epoch}/{args.epochs} mse={total / len(dataset):.4f}")
    return model


def train_patchtst(df_train: pd.DataFrame, args):
    model = PatchTST(
        h=args.h,
        input_size=args.input_size,
        patch_len=16,
        stride=8,
        hidden_size=256,
        linear_hidden_size=256,
        batch_size=args.batch_size,
        encoder_layers=4,
        n_heads=32,
        scaler_type="identity",
        loss=RMSE(),
        valid_loss=RMSE(),
        learning_rate=1e-4,
        max_steps=args.patchtst_steps,
        activation="ReLU",
        val_check_steps=50,
    )
    nf = NeuralForecast(models=[model], freq="5min")
    nf.fit(df=df_train, val_size=7858, verbose=False)
    return nf


def train_native_dlinear(df_train: pd.DataFrame, args):
    model = NativeDLinear(
        h=args.h,
        input_size=args.input_size,
        moving_avg_window=25,
        scaler_type="identity",
        loss=RMSE(),
        learning_rate=1e-3,
        max_steps=1000,
        val_check_steps=50,
    )
    nf = NeuralForecast(models=[model], freq="5min")
    nf.fit(df=df_train, val_size=7858, verbose=False)
    return nf


def summarize(pred_by_core: Dict[int, np.ndarray], true_by_core: Dict[int, np.ndarray],
              name: str, latency_ms: np.ndarray, h: int) -> dict:
    pred_all = np.concatenate([pred_by_core[u] for u in sorted(pred_by_core)])
    true_all = np.concatenate([true_by_core[u] for u in sorted(true_by_core)])

    diff = pred_all - true_all
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    y_mean = float(np.mean(true_all))
    ss_res = float(np.sum(diff ** 2))
    ss_tot = float(np.sum((true_all - y_mean) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    per_horizon = []
    for k in range(h):
        pk = np.concatenate([pred_by_core[u][k::h] for u in sorted(pred_by_core)])
        tk = np.concatenate([true_by_core[u][k::h] for u in sorted(true_by_core)])
        dk = pk - tk
        per_horizon.append({
            "horizon": k + 1,
            "rmse": float(np.sqrt(np.mean(dk ** 2))),
            "mae": float(np.mean(np.abs(dk))),
        })

    lat = np.asarray(latency_ms, dtype=np.float64)
    return {
        "model": name,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "rmse_pct": rmse / y_mean * 100.0,
        "mae_pct": mae / y_mean * 100.0,
        "y_mean": y_mean,
        "n_samples": int(pred_all.size),
        "latency_ms": {
            "mean": float(lat.mean()),
            "median": float(np.median(lat)),
            "p95": float(np.percentile(lat, 95)),
            "max": float(lat.max()),
        },
        "per_horizon": per_horizon,
    }


def evaluate_dlinear(model: DLinear, df_train: pd.DataFrame, df_test: pd.DataFrame,
                     args, device) -> dict:
    train_arrays = core_arrays(df_train)
    test_arrays = core_arrays(df_test)
    pred_by_core = {}
    true_by_core = {}
    latency_ms = []
    model.eval()
    with torch.no_grad():
        for uid in sorted(train_arrays):
            context = list(train_arrays[uid])
            test_y = test_arrays[uid]
            preds, trues = [], []
            for t0 in range(0, len(test_y), args.h):
                win = np.asarray(context[-args.input_size:], dtype=np.float32)
                xt = torch.from_numpy(win).unsqueeze(0).to(device)
                t_start = time.perf_counter()
                pred = model(xt)
                latency_ms.append((time.perf_counter() - t_start) * 1000.0)
                pred = pred.squeeze(0).cpu().numpy()
                chunk = test_y[t0 : t0 + args.h]
                if len(chunk) == args.h:
                    preds.append(pred)
                    trues.append(chunk)
                context.extend(chunk.tolist())
            pred_by_core[uid] = np.concatenate(preds)
            true_by_core[uid] = np.concatenate(trues)
    return summarize(pred_by_core, true_by_core, "DLinear", latency_ms, args.h)


def evaluate_neuralforecast(nf: NeuralForecast, model_col: str, df_train: pd.DataFrame,
                            df_test: pd.DataFrame, args, name: str) -> dict:
    test_arrays = core_arrays(df_test)
    uids = sorted(test_arrays)
    n_per_core = len(test_arrays[uids[0]])
    pred_by_core = {u: [] for u in uids}
    true_by_core = {u: [] for u in uids}
    latency_ms = []

    known = df_train.copy()
    for t0 in range(0, n_per_core, args.h):
        tail_times = known["ds"].unique()[-args.input_size:]
        tail_df = known[known["ds"].isin(tail_times)]
        t_start = time.perf_counter()
        fc = nf.predict(df=tail_df, verbose=False)
        latency_ms.append((time.perf_counter() - t_start) * 1000.0)

        next_times = fc["ds"].unique()
        actual = df_test[df_test["ds"].isin(next_times)]
        for uid in uids:
            f = fc[fc["unique_id"] == uid].sort_values("ds")
            a = actual[actual["unique_id"] == uid].sort_values("ds")
            k = min(len(f), len(a))
            if k == args.h:
                pred_by_core[uid].append(f[model_col].to_numpy()[:k])
                true_by_core[uid].append(a["y"].to_numpy()[:k])

        known = pd.concat([known, actual], ignore_index=True)
        known = known.sort_values(["ds", "unique_id"]).reset_index(drop=True)

    for uid in uids:
        pred_by_core[uid] = np.concatenate(pred_by_core[uid])
        true_by_core[uid] = np.concatenate(true_by_core[uid])
    return summarize(pred_by_core, true_by_core, name, latency_ms, args.h)


def count_params(model) -> int:
    try:
        return sum(p.numel() for p in model.parameters())
    except AttributeError:
        pass
    for attr in ("model", "models"):
        try:
            obj = getattr(model, attr)
            if isinstance(obj, (list, tuple)):
                return sum(p.numel() for m in obj for p in m.model.parameters())
            return sum(p.numel() for p in obj.parameters())
        except AttributeError:
            continue
    return 0


def print_table(results: dict) -> None:
    print("\n" + "=" * 78)
    print(f"Table V + VI - DLinear vs PatchTST (Bitbrains, dt=5min, h={results['h']})")
    print("=" * 78)
    header = (f"{'Model':<16}{'RMSE':>9}{'MAE':>9}{'R2':>9}{'RMSE%':>9}"
              f"{'MAE%':>9}{'lat mean':>11}{'lat p95':>11}{'params':>10}")
    print(header)
    print("-" * 78)
    for m in results["models"]:
        print(f"{m['model']:<16}{m['rmse']:>9.4f}{m['mae']:>9.4f}{m['r2']:>9.4f}"
              f"{m['rmse_pct']:>9.3f}{m['mae_pct']:>9.3f}"
              f"{m['latency_ms']['mean']:>9.2f}ms{m['latency_ms']['p95']:>8.2f}ms"
              f"{m['num_params']:>10}")
    print("=" * 78)


def run_model(args, device) -> dict:
    df = load_bitbrains(args.bitbrains_path)
    df_train, df_test = split_dataset(df)
    models = []
    torch.set_float32_matmul_precision("medium")

    if args.model in ("dlinear", "both"):
        dlinear = train_dlinear(df_train, args, device)
        res = evaluate_dlinear(dlinear, df_train, df_test, args, device)
        res["params"] = dlinear
        models.append(res)
        print(f"[dlinear] done: rmse={res['rmse']:.4f} mae={res['mae']:.4f} r2={res['r2']:.4f}")
        if args.dlinear_impl in ("native", "both"):
            nf = train_native_dlinear(df_train, args)
            res_n = evaluate_neuralforecast(nf, "DLinear", df_train, df_test, args,
                                            "DLinear (nf)")
            res_n["params"] = nf
            models.append(res_n)
            print(f"[dlinear-native] done: rmse={res_n['rmse']:.4f} mae={res_n['mae']:.4f}")

    if args.model in ("patchtst", "both"):
        nf = train_patchtst(df_train, args)
        res = evaluate_neuralforecast(nf, "PatchTST", df_train, df_test, args, "PatchTST")
        res["params"] = nf
        models.append(res)
        print(f"[patchtst] done: rmse={res['rmse']:.4f} mae={res['mae']:.4f} r2={res['r2']:.4f}")

    for m in models:
        m["num_params"] = count_params(m.pop("params"))

    result = {
        "h": args.h,
        "input_size": args.input_size,
        "device": str(device),
        "models": models,
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["dlinear", "patchtst", "both"], default="both")
    parser.add_argument("--h", type=int, default=6)
    parser.add_argument("--input-size", type=int, default=48)
    parser.add_argument("--moving-avg", type=int, default=25)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patchtst-steps", type=int, default=100)
    parser.add_argument("--dlinear-impl", choices=["custom", "native", "both"], default="custom")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--bitbrains-path", default=BITBRAINS_PATH)
    parser.add_argument("--results", default=os.path.join("experiments", "benchmark_results.json"))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    result = run_model(args, device)
    print_table(result)
    with open(args.results, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[saved] {args.results}")


if __name__ == "__main__":
    main()