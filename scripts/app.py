import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import cvxpy as cp

# --- Backend Functions ---
def mean_variance_optimizer(expected_returns, cov_matrix, lambda_risk=0.5, allow_shorting=False):
    n_assets = len(expected_returns)
    w = cp.Variable(n_assets)
    constraints = [cp.sum(w) == 1]
    if not allow_shorting:
        constraints.append(w >= 0)
    
    objective = cp.Maximize(w @ expected_returns - lambda_risk * cp.quad_form(w, cov_matrix))
    problem = cp.Problem(objective, constraints)
    problem.solve()
    
    optimal_weights = w.value.round(4)
    portfolio_return = (optimal_weights @ expected_returns).round(4)
    portfolio_volatility = np.sqrt(optimal_weights @ cov_matrix @ optimal_weights).round(4)
    
    return optimal_weights, portfolio_return, portfolio_volatility

def monte_carlo_simulation(expected_returns, cov_matrix, n_simulations=5000, allow_shorting=False):
    n_assets = len(expected_returns)
    mc_returns, mc_volatilities = [], []
    
    for _ in range(n_simulations):
        if allow_shorting:
            weights = np.random.randn(n_assets)
            weights /= weights.sum()
        else:
            weights = np.random.dirichlet(np.ones(n_assets))
        
        ret = weights @ expected_returns
        vol = np.sqrt(weights @ cov_matrix @ weights)
        mc_returns.append(ret)
        mc_volatilities.append(vol)
    
    return mc_returns, mc_volatilities

def calculate_efficient_frontier(expected_returns, cov_matrix, allow_shorting=False):
    n_assets = len(expected_returns)
    target_returns = np.linspace(np.min(expected_returns), np.max(expected_returns), 50)
    frontier_volatilities = []
    
    for r in target_returns:
        w = cp.Variable(n_assets)
        constraints = [cp.sum(w) == 1, w @ expected_returns >= r]
        if not allow_shorting:
            constraints.append(w >= 0)
        
        prob = cp.Problem(cp.Minimize(cp.quad_form(w, cov_matrix)), constraints)
        prob.solve()
        frontier_volatilities.append(np.sqrt(prob.value))
    
    return target_returns, frontier_volatilities

# --- Streamlit UI ---
st.title("Portfolio Optimizer with Risk Aversion")
st.markdown("""
Optimize your portfolio using **Mean-Variance Optimization (MVO)** and **Monte Carlo simulations**.
""")

# --- Sidebar Inputs ---
st.sidebar.header("Input Parameters")

# 1. Number of assets and names
n_assets = st.sidebar.number_input("Number of Assets", min_value=2, max_value=10, value=3)
asset_names = [st.sidebar.text_input(f"Asset {i+1} Name", value=f"Asset {i+1}") for i in range(n_assets)]

# 2. Expected returns and volatilities
expected_returns = []
volatilities = []
for i in range(n_assets):
    col1, col2 = st.sidebar.columns(2)
    with col1:
        mu = st.number_input(f"Expected Return (%) {asset_names[i]}", min_value=-50.0, max_value=50.0, value=5.0 + i*2)
        expected_returns.append(mu / 100)
    with col2:
        sigma = st.number_input(f"Volatility (%) {asset_names[i]}", min_value=0.0, max_value=100.0, value=10.0 + i*5)
        volatilities.append(sigma / 100)

# 3. Correlation matrix
st.sidebar.subheader("Correlations")
correlations = np.eye(n_assets)
for i in range(n_assets):
    for j in range(i+1, n_assets):
        corr = st.sidebar.slider(
            f"{asset_names[i]} vs {asset_names[j]}",
            min_value=-1.0, max_value=1.0, value=0.0
        )
        correlations[i][j] = corr
        correlations[j][i] = corr

# 4. Risk aversion and Monte Carlo settings
lambda_risk = st.sidebar.slider(
    "Risk Aversion (λ)", 
    min_value=0.1, max_value=5.0, value=1.0, step=0.1,
    help="Higher λ = more conservative portfolio"
)

n_simulations = st.sidebar.number_input(
    "Number of Monte Carlo Simulations", 
    min_value=100, max_value=10000, value=5000
)

allow_shorting = st.sidebar.checkbox("Allow Short-Selling", False)

# --- Compute Results ---
expected_returns = np.array(expected_returns)
cov_matrix = np.outer(volatilities, volatilities) * correlations

optimal_weights, portfolio_return, portfolio_volatility = mean_variance_optimizer(
    expected_returns, cov_matrix, lambda_risk, allow_shorting
)

mc_returns, mc_volatilities = monte_carlo_simulation(
    expected_returns, cov_matrix, n_simulations, allow_shorting
)

target_returns, frontier_volatilities = calculate_efficient_frontier(
    expected_returns, cov_matrix, allow_shorting
)

# --- Display Results ---
st.header("Optimal Portfolio Allocation")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Weights")
    allocation = pd.DataFrame({
        "Asset": asset_names,
        "Weight": optimal_weights,
        "Allocation (%)": (optimal_weights * 100).round(2)
    })
    st.dataframe(allocation)

with col2:
    st.subheader("Portfolio Metrics")
    metrics = pd.DataFrame({
        "Metric": ["Expected Return", "Volatility", "Risk Aversion (λ)", "MC Simulations"],
        "Value": [
            f"{portfolio_return*100:.2f}%", 
            f"{portfolio_volatility*100:.2f}%",
            lambda_risk,
            n_simulations
        ]
    })
    st.dataframe(metrics)

# Plot
st.header("Efficient Frontier & Monte Carlo Simulation")
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(mc_volatilities, mc_returns, c='gray', alpha=0.3, label=f"Random Portfolios (n={n_simulations})")
ax.plot(frontier_volatilities, target_returns, 'b-', linewidth=2, label="Efficient Frontier")
ax.scatter(portfolio_volatility, portfolio_return, c='red', s=200, marker='*', label=f"Optimal Portfolio (λ={lambda_risk})")
ax.set_xlabel("Volatility (σ)")
ax.set_ylabel("Expected Return (μ)")
ax.set_title("Portfolio Optimization")
ax.legend()
ax.grid()
st.pyplot(fig)

# Rebalancing Calculator
st.header("Rebalancing Calculator")
initial_capital = st.number_input("Total Investment Amount ($)", min_value=1000, value=100000)
allocation_amounts = optimal_weights * initial_capital
st.write("Recommended Allocation:")
allocation_df = pd.DataFrame({
    "Asset": asset_names,
    "Amount ($)": allocation_amounts.round(2),
    "Weight (%)": (optimal_weights * 100).round(2)
})
st.dataframe(allocation_df)