# Real-Data Optimal Execution

This part of the project extends the synthetic Almgren–Chriss benchmark to a real limit-order-book setting and focuses on optimal liquidation over short intraday episodes. The real-data contribution is a causal pipeline built from stock order-book and trade data, followed by a simple tabular reinforcement-learning / backward-DP style policy evaluation procedure. :contentReference[oaicite:0]{index=0}

## Overview

The real-data task uses **4,705,051 order-book updates** and **259,694 trades** across **3 stocks** and **5 trading days**. The objective is to liquidate inventory over a short fixed horizon while balancing immediate execution against the option value of waiting or posting passively. :contentReference[oaicite:1]{index=1}

The pipeline is intentionally conservative:

- it reconstructs the market **causally**,
- computes features using **past information only**,
- fits discretization **inside training folds only** to avoid leakage,
- and uses a **compact state space** because the dataset becomes sparse once converted into execution episodes. 

## Real-data pipeline

Each stock-day stream is processed as follows:

1. reconstruct a **level-1 order book** by forward-filling best bid and ask quotes,
2. merge order-book updates and trades into a single causal event stream,
3. compute market features using only information available up to the current time,
4. split the stream into **120-second episodes**,
5. define **8 decision steps** of **15 seconds** each,
6. train the execution model using **training days only**. :contentReference[oaicite:3]{index=3}

## State representation

The final selected state contains:

- **stock identity**
- **remaining inventory**
- discretized **spread**
- discretized **order-book imbalance**
- discretized **one-step return** :contentReference[oaicite:4]{index=4}

These features were selected through **nested cross-validation**. The project first evaluated smaller and larger candidate state spaces, starting from a private state with only stock and inventory, then progressively adding market variables. The best validation performance was obtained with **spread + imbalance + return**, which became the fixed configuration for the final outer-fold evaluation. :contentReference[oaicite:5]{index=5}

## Action space

At each decision step, the agent can choose among a small set of execution actions:

- **wait**
- send a **market order** for a fraction of the remaining inventory
- post a **passive ask** for a fraction of the remaining inventory :contentReference[oaicite:6]{index=6}

Execution is modeled simply:

- market orders execute immediately at the current bid,
- passive asks execute only when the next book move is favorable enough,
- any remaining inventory is forcibly liquidated at the end of the episode. :contentReference[oaicite:7]{index=7}

## Learning / cost estimation

Rather than training a deep RL agent, the real-data part uses an empirical tabular cost model with a backward update:

\[
C_b(s_t, a_t) = \hat{c}(s_t, a_t) + \min_{a'} C_b(s_{t+1}, a')
\]

where \(\hat{c}(s_t, a_t)\) is the mean observed cost for a visited state-action pair. :contentReference[oaicite:8]{index=8}

Because many exact states are rarely visited, the project uses a **hierarchical backoff** strategy:

1. exact state,
2. stock + inventory,
3. inventory only,
4. terminal fallback. :contentReference[oaicite:9]{index=9}

This keeps the method usable even on very small folds. :contentReference[oaicite:10]{index=10}

## Evaluation protocol

The final model is evaluated with **outer-fold testing**:

- in each fold, **one day** is used for testing,
- the remaining days are used for training,
- feature discretization is fit **only on the training split**. :contentReference[oaicite:11]{index=11}

This gives a clean out-of-sample estimate on real data. :contentReference[oaicite:12]{index=12}

## Results

Average implementation shortfall on the real-data task:

| Strategy | Mean cost (bps) | Std |
|---|---:|---:|
| RL policy | **1.34** | 0.41 |
| Immediate Market | 1.75 | 0.11 |
| TWAP (market orders) | 1.75 | 0.41 |

The learned policy improves on the main execution baselines on average across folds. :contentReference[oaicite:13]{index=13}

## Key takeaway

The main constraint in this project is **data sparsity**, not algorithmic sophistication. Once the event stream is converted into short execution episodes, the number of visited state-action pairs becomes limited. Richer state spaces often hurt reliability instead of helping. The project’s main conclusion is that, in this setting, **compact state design** and **better market simulation** matter more than adding more tabular complexity or switching to a more advanced RL algorithm. 

## Limitations

The real-data simulator remains simplified:

- passive fills are approximated from the **next book state**,
- there is no explicit modeling of **queue position**,
- no explicit **order priority**,
- and no detailed **slippage** model. :contentReference[oaicite:15]{index=15}

This likely limits the achievable performance more than the learning procedure itself. :contentReference[oaicite:16]{index=16}

## Repository focus

The real-data part of the repository is centered on four components:

- **data reconstruction** from raw book/trade files,
- **causal feature engineering**,
- **episode generation** for finite-horizon execution,
- **tabular backward cost estimation and evaluation**. This corresponds to the project’s stated main contribution. 
