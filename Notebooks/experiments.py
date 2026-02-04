# %%
import numpy as np
from scipy.stats import norm, qmc
import matplotlib.pyplot as plot
import torch
import torch.nn as nn

# %% [markdown]
# ## Data Generation using Analytical Black scholes
# 
# Generate synthetic training data using realistic distributions that reflect the real market characteristics while maininting the braod surface coverage. We are using Sobol for the data generation process for uniformity and spacefilling 
# 
# Domain for data generation:
# - $x=(ln(S/K),T,r,\sigma)$, 
# - $ln(S/K) \in [-1.5, 1.5]$, 
# - $T \in [0.01, 2.0]$ in years, 
# - $r \in [0.04]$ fixed, can vary later, 
# - $\sigma\in[0.05,0.90].$ 

# %% [markdown]
# ## Data Sampling

# %%
sampler = qmc.Sobol(d=1, scramble=True, seed=42)
sample = sampler.random_base2(m=10)
log_sk = qmc.scale(sample, -1.5, 1.5)
sk = qmc.scale(sample, 0.22, 4.48) 
sk = np.log(sk)
plot.hist(log_sk, bins=50, alpha=0.6, color='red')
plot.hist(sk, bins=50, alpha=0.6)
plot.show()

# %%
sampler = qmc.Sobol(d=1, scramble=True, seed=42)
sample = sampler.random_base2(m=10)
l_roott = np.sqrt(0.01)
u_roott = np.sqrt(2)
troot = qmc.scale(sample, l_roott, u_roott)
t = qmc.scale(sample, 0.01, 2)
T = troot**2
plot.hist(T, bins=50, alpha=0.6, color='red')
plot.hist(t, bins=50, alpha=0.6)
plot.show()

# %% [markdown]
# ## Blacksholes Implementation 

# %%
# blackscholes implementation 
class BlackScholesPricer: 
    def __init__(self, r: float):
        self.r = r                                            # since r is a model parameter we initialize this seperately      

    def price(self, S, K, T, sigma, option_type='C'):
        """
        Calculates the blackscholes option price for call and put options 
        Args:
            S (float): Stock price       
            K (float): Strike price                
            T (float): Time to maturity in years 
            r (float): risk free rate 
            sigma (float): volatility of the underlying 
        Returns:
            option price (float)
        Raises:
            ValueError: if sigma < 0 and T <= 0 
        """
        # input casting and vectorize
        S, K, T, sigma = map(np.asarray, (S, K, T, sigma))
        
        # Input validation 
        if np.any(sigma < 0):
            raise ValueError("Volatility cannot be negative")
        if np.any(T <= 0):
            raise ValueError("Expiry must be > 0")
        
        # black scholes variables 
        sqrt_T = np.sqrt(T)
        d1 = (np.log(S / K) + (self.r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        
        # dic factor
        disc = np.exp(-self.r * T)
        
        # black scholes pricing formula 
        if option_type == 'C':
            price = S * norm.cdf(d1) - K * disc * norm.cdf(d2) 
        elif option_type == 'P':
            price = K * disc * norm.cdf(-d2) - S * norm.cdf(-d1) 
        else:
            raise ValueError(f"option type must be 'C' or 'P got '{option_type}'")
            
        return price

# %% [markdown]
# #### Homogenity

# %%
# since the blacksholes is homogenous of degree one we can price options in the unit of K 
class LogMoneynessPricer:
    
    def __init__(self, bs_pricer: BlackScholesPricer):
        self.bs_pricer = bs_pricer
        
    def price(self, M, T, sigma, option_type='C'):
        M, T, sigma = map(np.asarray, (M, T, sigma))
        
        K = 1.0 # price in units of K 
        S = np.exp(M) * K 
        
        C = self.bs_pricer.price(S, K, T, sigma, option_type)
        return C / K 

# %% [markdown]
# #### Data generator

# %%
# dataGenerator.py
# Sobol sampler 
class SobolSampler:
    def __init__(self, l_bounds, u_bounds, seed=42):
        self.l_bounds = l_bounds 
        self.u_bounds = u_bounds
        self.sampler = qmc.Sobol(d=3, scramble=True, seed=seed)
        
    def sample(self, m):
        unit_cube = self.sampler.random_base2(m=m)
        return qmc.scale(unit_cube, self.l_bounds, self.u_bounds)

# Data generator 
class BSDataGenerator:
    def __init__(self, sampler: SobolSampler, pricer: LogMoneynessPricer):
        self.sampler = sampler 
        self.pricer = pricer 
    
    def generate(self, m, option_type='C'):
        sample = self.sampler.sample(m)
        
        M, sqrt_T, sigma = sample.T 
        T = sqrt_T**2 
        
        X = np.column_stack([M, T, sigma])
        y = self.pricer.price(M, T, sigma, option_type)
        return X, y 

# %% [markdown]
# #### Sampling 

# %%
bs = BlackScholesPricer(r=0.04)
l_bounds = [-1.5, np.sqrt(0.01), 0.05]
u_bounds = [1.5, np.sqrt(2), 0.60]
sample = SobolSampler(l_bounds, u_bounds)
logm_pricer = LogMoneynessPricer(bs)

gen = BSDataGenerator(sample, logm_pricer)
X, y = gen.generate(m=14)

# %%
feature = torch.tensor(X, dtype=torch.float32)
target = torch.tensor(y, dtype=torch.float32)
print(feature.shape)
print(target.shape)

# %%
print(feature.dtype)


# %% [markdown]
# ### Training Baseline Model 

# %%
# layer 1
torch.manual_seed(42)
layer1 = nn.Linear(in_features=3, out_features=64)
output1 = layer1(feature)                       
a = nn.ELU()                     # elu activation
l1 = a(output1)
print(f"layer 1: {l1.shape}")

# layer 2 
layer2 = nn.Linear(in_features=64, out_features=64)
output2 = layer2(l1)                       
a = nn.ELU()                     # elu activation
l2 = a(output2)
print(f"layer 2: {l2.shape}")

# layer 3 
layer3 = nn.Linear(in_features=64, out_features=1)
output3 = layer3(l2)             # no activation at output layer
print(f"layer 3: {output3.shape}")

# mse loss calculation mse 
output3 = output3.squeeze()
output3.shape
mse = 1/len(output3)*(sum((output3 - target)**2))
print(f"Manual MSE: {mse}") 

# mse 
loss_fn = nn.MSELoss()
mse = loss_fn(output3, target)
print(f"NN.MSE: {mse}")

# parameter 
list(layer1.parameters())
for p in layer1.parameters():
    print(p.shape)
all_params = list(layer1.parameters()) + list(layer2.parameters()) + list(layer3.parameters())
len(all_params)

# optimizer
optimizer = torch.optim.Adam(all_params, lr=0.001)
optimizer.zero_grad()                   # clear the old gradients 
mse.backward()
optimizer.step()

# %%
# Forward pass 2
output1 = layer1(feature)
l1 = a(output1)
output2 = layer2(l1)
l2 = a(output2)
output3 = layer3(l2).squeeze()

# New loss
new_mse = loss_fn(output3, target)
print(f"New MSE: {new_mse}")

# %%
for epoch in range(1000):
    # forward pass 
    output1 = layer1(feature)
    l1 = a(output1)
    output2 = layer2(l1)
    l2 = a(output2)
    output3 = layer3(l2).squeeze()
    
    # compute loss 
    mse = loss_fn(output3, target)
    
    # backward pass 
    optimizer.zero_grad()                   # clear the old gradients 
    mse.backward()

    # update weights
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, loss: {mse.item()}")

# %%
# Get predictions, no gradient for eval
with torch.no_grad():                            # we use no_grad() to disable the gradient trackiing during evaluation this skip the graph building by pytorch  
    output1 = layer1(feature)                    
    l1 = a(output1)
    output2 = layer2(l1)
    l2 = a(output2)
    predictions = layer3(l2).squeeze()

# Compare
print("Predicted | Actual")
for i in range(10):
    print(f"{predictions[i].item():.4f} vs {target[i].item():.4f}") 

# %% [markdown]
# ### Observation 
# Model produces small negative values for OTM options, consider adding softplus or ReLu output activation or post-processing 

# %%
## refactor 
class OptionPricer(nn.Module):
    def __init__(self):
        super().__init__()
        # layers and activation  
        self.layer1 = nn.Linear(in_features=3, out_features=64)     # layer 1
        self.activation = nn.ELU()                                      # activation 1               
        self.layer2 = nn.Linear(in_features=64, out_features=64)     # layer 2                             
        self.layer3 = nn.Linear(in_features=64, out_features=1)      # layer 3
        
    def forward(self, x):
        # pass features through layers and return output 
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.activation(x)
        x = self.layer3(x)
        return x

# %%
torch.manual_seed(42)
model = OptionPricer()

# Test for forward pass
output = model(feature)

# no of params and output shape
print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
print(f"Output shape: {output.shape}")

# %%
torch.manual_seed(42)
model = OptionPricer()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

for epoch in range(1000):
    # forward pass 
    output = model(feature).squeeze()
    
    # compute loss 
    mse = loss_fn(output, target)
    
    #backward and update weight 
    optimizer.zero_grad()
    mse.backward()
    optimizer.step()
    
    if epoch % 50 == 0:
        print(f"Epoch: {epoch}, loss: {mse.item():.6f}")    

# %%
# spliting the dataset, 
# since the sobol sequnce generate quasi-random and uniform, low-descrepancy data, we are spliting randomly for better convergence 

n = len(feature)
indices = torch.randperm(n) # create permutation for 0 to n-1 

# split point 
train_end = int(0.70 * n)
val_end = int(0.85 * n)

# list of index 
train_idx = indices[:train_end]
val_idx = indices[train_end:val_end]
test_idx = indices[val_end:]

# %%
# train, validation and test split 
# feature space 
x_train = feature[train_idx]
x_val = feature[val_idx]
x_test = feature[test_idx]

# predictor space 
y_train = target[train_idx]
y_val = target[val_idx]
y_test = target[test_idx]
print(f" x: train shape={x_train.shape}, val shape={x_val.shape}, test shape={x_test.shape}")
print(f" y: train shape={y_train.shape}, val shape={y_val.shape}, test shape={y_test.shape}")

# %% [markdown]
# #### test train and val split 

# %%
torch.manual_seed(42)
model = OptionPricer()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

for epoch in range(1000):
    # training 
    model.train()                     # this sets the dropout and batch stats -> use BatchNorm
    output = model(x_train).squeeze()
    train_loss = loss_fn(output, y_train)

    #backward and update weight 
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()
    
    # validation 
    model.eval()
    with torch.no_grad():
        val_output = model(x_val).squeeze()
        val_loss = loss_fn(val_output, y_val)
    
    if epoch % 50 == 0:
        print(f"Epoch: {epoch}, loss:{train_loss.item():.6f}, val loss:{val_loss.item():.6f}") 

# %% [markdown]
# #### observation: test and val decreasing together, model is not overfitting but its a small sample 

# %%
# model evaluation 
model.eval()
with torch.no_grad():
    test_output = model(x_test).squeeze()
    test_loss = loss_fn(test_output, y_test)
    print(f"Test MSE:{test_loss.item():.6f}") 

# %% [markdown]
# ### Gaussian weighted loss function

# %%
sigma_m = 0.25
beta = 2.0

M = x_train[:, 0]  # log moneyness
T = x_train[:, 1]  # time to expiry
M_val = x_val[:, 0]
T_val = x_val[:, 1]

weights_train = torch.exp(-(M**2) / (2 * sigma_m**2) - beta * T)
weights_val = torch.exp(-(M_val**2) / (2 * sigma_m**2) - beta * T_val)

# %%
print(f"shape: {weights_train.shape}, Min:{weights_train.min()}, Max:{weights_train.max()}") 
print(f"Mean: {weights_train.mean()}")
print(f"Median: {weights_train.median()}")
print(f"% above 0.1: {(weights_train > 0.1).float().mean() * 100:.1f}%")
print(f"% above 0.01: {(weights_train > 0.01).float().mean() * 100:.1f}%")
print(f"------------------------------------------------------------------------------------")
print(f"shape: {weights_val.shape}, Min:{weights_val.min()}, Max:{weights_val.max()}") 
print(f"Mean: {weights_val.mean()}")
print(f"Median: {weights_val.median()}")
print(f"% above 0.1: {(weights_val > 0.1).float().mean() * 100:.1f}%")
print(f"% above 0.01: {(weights_val > 0.01).float().mean() * 100:.1f}%")

# %%
def weighted_mse(predictions, targets, weights):
    # weighted loss function 
    sqaured_errors = (targets - predictions)**2
    wmse = torch.sum(weights * sqaured_errors) / torch.sum(weights)
    return wmse
    

# %%
model_weighted = OptionPricer()
optimizer = torch.optim.Adam(model_weighted.parameters(), lr=0.001)

for epoch in range(1000):
    # training 
    model_weighted.train()                     # this sets the dropout and batch stats -> use BatchNorm
    output = model_weighted(x_train).squeeze()
    train_loss = weighted_mse(output, y_train, weights_train)

    #backward and update weight 
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()
    
    # validation 
    model_weighted.eval()
    with torch.no_grad():
        val_output = model_weighted(x_val).squeeze()
        val_loss = weighted_mse(val_output, y_val, weights_val)
    
    if epoch % 50 == 0:
        print(f"Epoch: {epoch}, loss:{train_loss.item():.6f}, val loss:{val_loss.item():.6f}") 

# %%
# model comparision 
# standard mse model 
# model evaluation 
model_weighted.eval()
with torch.no_grad():
    test_output = model_weighted(x_test).squeeze()
    
    # weights for test set 
    M_test = x_test[:, 0]
    T_test  =x_test[:, 1]
    weights_test = torch.exp(-(M_test**2) / (2 * sigma_m**2) - beta * T_test)
    
    weighted_test_loss = weighted_mse(test_output, y_test, weights_test)
    test_loss = loss_fn(test_output, y_test)
    print(f"Test WMSE:{test_loss.item():.6f}") 

# %%
model_weighted.eval()
with torch.no_grad():
    pred = model_weighted(x_test).squeeze()
    
    standard_mse = nn.MSELoss()(pred, y_test)
    wmse = weighted_mse(pred, y_test, weights_test)
    
print(f"Weighted model - Standard MSE: {standard_mse.item():.6f}")
print(f"Weighted model - Weighted MSE: {wmse.item():.6f}")

# %% [markdown]
# #### Comparision 

# %%
# ATM region: |M| < 0.1
atm_mask = torch.abs(x_test[:, 0]) < 0.1

print(f"Number of ATM samples: {atm_mask.sum().item()} / {len(x_test)}")

with torch.no_grad():
    baseline_pred = model(x_test).squeeze()
    weighted_pred = model_weighted(x_test).squeeze()
    
    baseline_atm = nn.MSELoss()(baseline_pred[atm_mask], y_test[atm_mask])
    weighted_atm = nn.MSELoss()(weighted_pred[atm_mask], y_test[atm_mask])
    
print(f"ATM only MSE Baseline: {baseline_atm.item():.6f}")
print(f"ATM only MSE Weighted: {weighted_atm.item():.6f}")

# %% [markdown]
# # Real Data

# %%
pip install yfinance

# %%
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# %%
# SPY - American Options
ticker = yf.Ticker('SPY')
spot = ticker.history(period='1d')['Close'].iloc[-1]
available_expirations = ticker.options
print(available_expirations[:5])

all_options = []
for exp in available_expirations:
    try: 
        chain = ticker.option_chain(exp)
        calls = chain.calls.copy()
        calls['expiry'] = exp 
        all_options.append(calls)
        print(f"Expiry :{exp}: {len(calls)} calls")
    except Exception as e:
        print(f"failed {exp}: {e}")
        continue 

df = pd.concat(all_options, ignore_index=True)
print(f"\ntotal contract {len(df)}")

# %%
print(f"Raw contracts: {len(df)}")
print(df[['strike', 'bid', 'ask', 'impliedVolatility', 'openInterest']].head(10))
print(df[['strike', 'bid', 'ask', 'impliedVolatility', 'openInterest']].describe())

# %%
pip install datetime

# %% [markdown]
# #### Cleaned data for stale quotes

# %%
# cleaning 0 bid and offer 
print(f"raw: {len(df)}")
df_clean = df[ (df['bid'] > 0) | (df['openInterest'] > 0)] 

from datetime import datetime as dt

today = dt.now()
df['T'] = pd.to_datetime(df['expiry']).apply(lambda x: (x - today).days / 365)
df['is_0DTE'] = df['T'] < (1/365)  # less than 1 day
print(f"0DTE contracts: {df['is_0DTE'].sum()}")

# %%
df_clean = df[
    (df['ask'] > 0) &                    # Must have ask
    (df['impliedVolatility'] > 0.001) &  # Must have some IV
    (df['impliedVolatility'] < 5.0)      # IV < 500%, allow high for 0DTE
]

df_clean = df_clean[
    (df_clean['bid'] > 0) | 
    (df_clean['openInterest'] > 10)  # Some activity
]

df_clean = df_clean[df_clean['T'] > 0]
df_clean = df_clean[df_clean['bid'].notna()]
print(f"no of contract: {len(df_clean)}")

# %%
# Basic stats
print(f"Unique expiries: {df_clean['expiry'].nunique()}")
print(f"Strike range: [{df_clean['strike'].min()}, {df_clean['strike'].max()}]")

# IV distribution
print(f"\nIV stats:")
print(f"  Min: {df_clean['impliedVolatility'].min():.4f}")
print(f"  Max: {df_clean['impliedVolatility'].max():.4f}")
print(f"  Mean: {df_clean['impliedVolatility'].mean():.4f}")

# Check for NaNs
print(f"\nMissing values:")
print(df_clean[['strike', 'bid', 'ask', 'impliedVolatility']].isna().sum())

# ask should be > bid
bad_spread = (df_clean['ask'] < df_clean['bid']).sum()
print(f"\nBid > Ask (error): {bad_spread}")

# Zero checks
print(f"\nZero bids: {(df_clean['bid'] == 0).sum()}")
print(f"Zero asks: {(df_clean['ask'] == 0).sum()}")

# %%
df_clean

# %% [markdown]
# #### Calculating features

# %%
# Moneyness M = log(S/K)
df_clean['M'] = np.log(spot/df['strike'])
df_clean['M'].describe()

# time to exp in years 
# df_clean['T'] = pd.to_datetime(df['expiry']).apply((lambda x: (x - today).days / 365))
# df_clean

# 13 week tbill for market proxy 
tnx = yf.Ticker("^IRX")  # 13-week T-bill
r = tnx.history(period="1d")['Close'].iloc[-1] / 100  
print(f"Current risk-free rate: {r:.4f}")
df_clean['r'] = r
df_clean

# %%
print(f"T <= 0: {(df_clean['T'] <= 0).sum()}")
print(f"Total samples: {len(df_clean)}")
print(df_clean[['M', 'T', 'r', 'impliedVolatility']].isna().sum())

# %%
# features and target extract 
features = df_clean[['M', 'T', 'r']].to_numpy()
target = df_clean['impliedVolatility'].to_numpy()

# convert to tensor 
X = torch.tensor(features, dtype=torch.float32)
y = torch.tensor(target, dtype=torch.float32)

# %%
# Train/val/test split
n = len(X)
indices = torch.randperm(n)

train_end = int(0.7 * n)
val_end = int(0.85 * n)

X_train = X[indices[:train_end]]
y_train = y[indices[:train_end]]
X_val = X[indices[train_end:val_end]]
y_val = y[indices[train_end:val_end]]
X_test = X[indices[val_end:]]
y_test = y[indices[val_end:]]

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# %%
class VolSurfaceModel(nn.Module):
    def __init__(self):           
        super().__init__()
        self.layer1 = nn.Linear(3, 64)   # (M, T, r)
        self.activation = nn.ELU()
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, 1)   # output: implied vol
        self.softplus = nn.Softplus()    #  ensures positive IV
        
    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        x = self.layer3(x)
        x = self.softplus(x)
        return x

# %%
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")

# %%
print(df_clean[['M', 'T', 'r']].shape)

# %%
# Compute weights for Gaussian-weighted loss
sigma_m = 0.2
beta = 1.0

M_train = X_train[:, 0]
T_train = X_train[:, 1]
weights_train = torch.exp(-(M_train**2) / (2 * sigma_m**2) - beta * T_train)
weights_train = torch.clamp(weights_train, min=0.1)


M_val = X_val[:, 0]
T_val = X_val[:, 1]
weights_val = torch.exp(-(M_val**2) / (2 * sigma_m**2) - beta * T_val)
weights_val = torch.clamp(weights_val, min=0.1)


# Train two models: baseline and weighted
torch.manual_seed(42)
model_baseline = VolSurfaceModel()
optimizer_baseline = torch.optim.Adam(model_baseline.parameters(), lr=0.001)

torch.manual_seed(42)
model_weighted = VolSurfaceModel()
optimizer_weighted = torch.optim.Adam(model_weighted.parameters(), lr=0.001)

loss_fn = nn.MSELoss()

# Training loop - both models
for epoch in range(1000):
    # Baseline
    pred_b = model_baseline(X_train).squeeze()
    loss_b = loss_fn(pred_b, y_train)
    optimizer_baseline.zero_grad()
    loss_b.backward()
    optimizer_baseline.step()
    
    # Weighted
    pred_w = model_weighted(X_train).squeeze()
    loss_w = weighted_mse(pred_w, y_train, weights_train)
    optimizer_weighted.zero_grad()
    loss_w.backward()
    optimizer_weighted.step()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Baseline={loss_b.item():.6f}, Weighted={loss_w.item():.6f}")

# %%
# Test set evaluation
M_test = X_test[:, 0]
T_test = X_test[:, 1]
weights_test = torch.exp(-(M_test**2) / (2 * sigma_m**2) - beta * T_test)

# ATM mask: |M| < 0.1
atm_mask = torch.abs(M_test) < 0.1

print(f"ATM samples in test: {atm_mask.sum().item()} / {len(X_test)}")

model_baseline.eval()
model_weighted.eval()

with torch.no_grad():
    pred_baseline = model_baseline(X_test).squeeze()
    pred_weighted = model_weighted(X_test).squeeze()
    
    # Overall MSE
    mse_baseline = loss_fn(pred_baseline, y_test)
    mse_weighted = loss_fn(pred_weighted, y_test)
    
    # ATM-only MSE
    atm_baseline = loss_fn(pred_baseline[atm_mask], y_test[atm_mask])
    atm_weighted = loss_fn(pred_weighted[atm_mask], y_test[atm_mask])

print(f"\nOverall MSE  - Baseline: {mse_baseline.item():.6f}, Weighted: {mse_weighted.item():.6f}")
print(f"ATM-only MSE - Baseline: {atm_baseline.item():.6f}, Weighted: {atm_weighted.item():.6f}")

# %%
model_weighted.eval()
with torch.no_grad():
    preds = model_weighted(x_test).squeeze()
    
    # Check 1:Any negative IV predictions?
    neg_count = (preds < 0).sum().item()
    print(f"Negative IV predictions: {neg_count} / {len(preds)}")
    
    # Check 2: Any unrealistic IV (>500%)?
    high_count = (preds > 5.0).sum().item()
    print(f"IV > 500%: {high_count} / {len(preds)}")
    
    # Check 3: Prediction range
    print(f"Predicted IV range: [{preds.min():.4f}, {preds.max():.4f}]")
    print(f"Actual IV range: [{y_test.min():.4f}, {y_test.max():.4f}]")

# %%
# Checking if IV decrease then increase with moneyness: vol smile 
# Group by moneyness buckets and check mean predicted IV

M_test = X_test[:, 0]
buckets = [(-1, -0.2), (-0.2, -0.1), (-0.1, 0.1), (0.1, 0.2), (0.2, 1)]

print("Moneyness Bucket | Mean Predicted IV | Mean Actual IV")
print("-" * 55)

with torch.no_grad():
    preds = model_weighted(X_test).squeeze()
    
for low, high in buckets:
    mask = (M_test >= low) & (M_test < high)
    if mask.sum() > 0:
        pred_mean = preds[mask].mean().item()
        actual_mean = y_test[mask].mean().item()
        print(f"[{low:+.1f}, {high:+.1f})       | {pred_mean:.4f}            | {actual_mean:.4f}")

# %%
# Check calendar arbitrage sigma**2 * T should increase with T
# Group by similar moneyness, checking if variance increases with T

with torch.no_grad():
    preds = model_weighted(X_test).squeeze()

M_test = X_test[:, 0]
T_test = X_test[:, 1]

# Focus on ATM options (|M| < 0.1)
atm_mask = torch.abs(M_test) < 0.1

M_atm = M_test[atm_mask]
T_atm = T_test[atm_mask]
pred_atm = preds[atm_mask]
actual_atm = y_test[atm_mask]

# Compute total variance
pred_var = (pred_atm ** 2) * T_atm
actual_var = (actual_atm ** 2) * T_atm

# Sort by time and check if variance increases
sorted_idx = torch.argsort(T_atm)
T_sorted = T_atm[sorted_idx]
pred_var_sorted = pred_var[sorted_idx]
actual_var_sorted = actual_var[sorted_idx]

# Check for violations (variance decreasing)
pred_violations = (pred_var_sorted[1:] < pred_var_sorted[:-1]).sum().item()
actual_violations = (actual_var_sorted[1:] < actual_var_sorted[:-1]).sum().item()

print(f"ATM samples: {atm_mask.sum().item()}")
print(f"Calendar arbitrage violations (pred): {pred_violations}")
print(f"Calendar arbitrage violations (actual): {actual_violations}")

# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

M_test_np = X_test[:, 0].numpy()
T_test_np = X_test[:, 1].numpy()

with torch.no_grad():
    pred_np = model_weighted(X_test).squeeze().numpy()
actual_np = y_test.numpy()

# Plot 1: Predicted IV vs Moneyness
ax1 = axes[0]
ax1.scatter(M_test_np, pred_np, alpha=0.3, s=10, label='Predicted')
ax1.scatter(M_test_np, actual_np, alpha=0.3, s=10, label='Actual')
ax1.set_xlabel('Log Moneyness (M)')
ax1.set_ylabel('Implied Volatility')
ax1.set_title('IV vs Moneyness')
ax1.legend()

# Plot 2: Predicted vs Actual (scatter)
ax2 = axes[1]
ax2.scatter(actual_np, pred_np, alpha=0.3, s=10)
ax2.plot([0, 1.5], [0, 1.5], 'r--', label='Perfect fit')
ax2.set_xlabel('Actual IV')
ax2.set_ylabel('Predicted IV')
ax2.set_title('Predicted vs Actual')
ax2.legend()

# Plot 3: Error by Moneyness
ax3 = axes[2]
error = pred_np - actual_np
ax3.scatter(M_test_np, error, alpha=0.3, s=10)
ax3.axhline(y=0, color='r', linestyle='--')
ax3.set_xlabel('Log Moneyness (M)')
ax3.set_ylabel('Prediction Error (Pred - Actual)')
ax3.set_title('Error vs Moneyness')

plt.tight_layout()
plt.savefig('vol_surface_results.png', dpi=150)
plt.show()

print("Saved: vol_surface_results.png")

# %%
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(14, 5))

# Filter for cleaner visualization (remove extreme moneyness)
mask = (M_test_np > -0.5) & (M_test_np < 0.8) & (T_test_np < 1.5)

M_plot = M_test_np[mask]
T_plot = T_test_np[mask]
pred_plot = pred_np[mask]
actual_plot = actual_np[mask]

# Plot 1: Actual surface
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(M_plot, T_plot, actual_plot, c=actual_plot, cmap='viridis', s=10, alpha=0.6)
ax1.set_xlabel('Moneyness')
ax1.set_ylabel('Time to Expiry')
ax1.set_zlabel('IV')
ax1.set_title('Actual IV Surface')

# Plot 2: Predicted surface
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(M_plot, T_plot, pred_plot, c=pred_plot, cmap='viridis', s=10, alpha=0.6)
ax2.set_xlabel('Moneyness')
ax2.set_ylabel('Time to Expiry')
ax2.set_zlabel('IV')
ax2.set_title('Predicted IV Surface')

plt.tight_layout()
plt.savefig('vol_surface_3d.png', dpi=150)
plt.show()

# %%
# Proper calendar arbitrage check: same strike region, different expiry
# Group by narrow moneyness bands, then check variance across time

print("Calendar Arbitrage Check (Proper)")
print("=" * 70)

M_test = X_test[:, 0]
T_test = X_test[:, 1]

with torch.no_grad():
    preds = model_weighted(X_test).squeeze()

# Narrow ATM band
atm_mask = torch.abs(M_test) < 0.05  # tighter band

M_atm = M_test[atm_mask]
T_atm = T_test[atm_mask]
iv_atm = y_test[atm_mask]

# Get unique expiries and average IV at each
unique_T = torch.unique(T_atm)
unique_T_sorted, _ = torch.sort(unique_T)

print(f"Unique expiries in ATM band: {len(unique_T_sorted)}")
print(f"\n{'Expiry (T)':<12} {'Avg IV':<12} {'Total Var':<12}")
print("-" * 40)

variances = []
for t in unique_T_sorted:
    mask = T_atm == t
    avg_iv = iv_atm[mask].mean().item()
    total_var = (avg_iv ** 2) * t.item()
    variances.append(total_var)
    print(f"{t.item():<12.4f} {avg_iv:<12.4f} {total_var:<12.6f}")

# Check for violations
variances = torch.tensor(variances)
diffs = variances[1:] - variances[:-1]
violations = (diffs < 0).sum().item()

print("-" * 40)
print(f"\nCalendar violations: {violations} / {len(diffs)}")

# %%
# Check the magnitude of real violations
print("Actual Calendar Violations:")
print("-" * 60)
print(f"{'T_short':<10} {'T_long':<10} {'Var_short':<12} {'Var_long':<12} {'Gap':<10}")
print("-" * 60)

T_list = unique_T_sorted.tolist()
var_list = variances.tolist()

for i in range(len(diffs)):
    if diffs[i] < 0:
        gap = abs(diffs[i].item())
        # Estimate profit: gap in variance terms
        # Rough conversion: gap * 100 = basis points
        print(f"{T_list[i]:<10.4f} {T_list[i+1]:<10.4f} {var_list[i]:<12.6f} {var_list[i+1]:<12.6f} {gap:<10.6f}")

print("-" * 60)
print("\nTransaction cost estimate: ~0.001 variance units")
print("Violations > 0.001 might be tradeable (before other costs)")

# %%
import time

# Your model - single prediction
start = time.time()
for _ in range(10000):
    with torch.no_grad():
        pred = model_weighted(X_test[:1])
nn_time = time.time() - start

print(f"NN: 10,000 predictions in {nn_time:.4f} seconds")
print(f"Per prediction: {nn_time/10000*1000:.4f} ms")

# %%
# Delta = ∂Price/∂S, but we predict IV, so let's check ∂IV/∂M
X_test.requires_grad_(True)
pred = model_weighted(X_test)
pred.sum().backward()

delta_iv = X_test.grad[:, 0]  # ∂IV/∂M
print(f"Mean ∂IV/∂M: {delta_iv.mean():.4f}")
print(f"Stable (no NaN/Inf): {torch.isfinite(delta_iv).all()}")


