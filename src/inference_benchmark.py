import time 
import json
import numpy as np
import torch 
from pathlib import Path

ROOT = Path.cwd().parent.resolve()
MODEL_DIR = ROOT / "src" / "models"

# features consruction 
def feature_cons(df, features):
    X = df[features].to_numpy(np.float32) 
    return X

# Z-score normalize
def normalize(X, mu, sd):
    return( X - mu) / sd 

# gaussian weights 
def gaussian_weights(
    df,
    lam,
    beta,     
    M_col,
    T_col,
    normalize_mean,
    w_floor_itm
):
    M = df[M_col].to_numpy(float)
    T = df[T_col].to_numpy(float)

    w = np.exp(-(M**2)/(2*lam**2) - beta*T)

    # snip floor for deep -in money 
    itM = M > 0.05
    w[itM] = np.maximum(w[itM], w_floor_itm)

    w = w.astype(np.float32)
    w /= (np.mean(w) + 1e-8)

    if normalize_mean:
        w /= (np.mean(w) + 1e-8)

    return w

def return_gaussian_weights(df, gw_params):
    return gaussian_weights(
        df,
        lam            = gw_params["lam"],
        beta           = gw_params["beta"],
        w_floor_itm    = gw_params["w_floor_itm"],
        M_col          = "M",
        T_col          = "t_ann",
        normalize_mean = True,
    )


# convert to tensor 
def to_tensor(X_z):
    return torch.tensor(X_z)


# forward pass without gradients 
def forward(model, x_t):
    with torch.no_grad():
        return model(x_t)


# tranform output 
def output_transform(pred):
    return pred.numpy().flatten()

def full_pipeline(df, model, mu, sd, features, gw_params):
    X      = feature_cons(df, features)
    X_z    = normalize(X, mu, sd)
    w      = return_gaussian_weights(df, gw_params)
    x_t    = to_tensor(X_z)
    pred   = forward(model, x_t)
    prices = output_transform(pred)
    return prices, w


def benchmark(df, model, mu, sd, features, gw_params,
              n_warmup=50, n_runs=500, label=""):
    torch.set_num_threads(1)

    # warmup
    for _ in range(n_warmup):
        X      = df[features].to_numpy(np.float32)
        X_z    = (X - mu) / sd
        w      = return_gaussian_weights(df, gw_params)  
        x_t    = torch.tensor(X_z)
        with torch.no_grad():
            pred = model(x_t)

    # timed
    latencies_ms = []
    for _ in range(n_runs):
        t0 = time.perf_counter()

        X      = df[features].to_numpy(np.float32)
        X_z    = (X - mu) / sd
        w      = return_gaussian_weights(df, gw_params)   
        x_t    = torch.tensor(X_z)
        with torch.no_grad():
            pred = model(x_t)
        prices = pred.numpy().flatten()

        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000)

    lat = np.array(latencies_ms)

    print(f"\n{'='*45}")
    print(f"  {label}")
    print(f"{'='*45}")
    print(f"  Mean        : {np.mean(lat):.4f} ms")
    print(f"  p50         : {np.percentile(lat, 50):.4f} ms")
    print(f"  p99         : {np.percentile(lat, 99):.4f} ms")
    print(f"  Throughput  : {len(df) / np.mean(lat) * 1000:,.0f} options/sec")
    print(f"{'='*45}")
