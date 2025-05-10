import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from tqdm import tqdm  # For progress bars (optional)

def portfolio_optimizer(
    expected_returns, 
    covariance_matrix, 
    lambda_risk=0.5, 
    n_simulations=1000,
    allow_shorting=False,
    plot_frontier=True,
    plot_monte_carlo=True
):
    """
    Enhanced MVO with Monte Carlo simulation for robustness testing.
    
    Parameters:
    -----------
    expected_returns : np.ndarray
        Vector of expected returns for each asset.
    covariance_matrix : np.ndarray
        Covariance matrix of asset returns.
    lambda_risk : float, optional (default=0.5)
        Risk aversion parameter (higher = more conservative).
    n_simulations : int, optional (default=1000)
        Number of Monte Carlo simulations.
    allow_shorting : bool, optional (default=False)
        If True, allows negative weights (short-selling).
    plot_frontier : bool, optional (default=True)
        Whether to plot the efficient frontier.
    plot_monte_carlo : bool, optional (default=True)
        Whether to plot Monte Carlo simulated portfolios.
    
    Returns:
    --------
    dict
        Optimal weights, expected return, volatility, and simulation results.
    """
    # Input validation
    assert len(expected_returns) == covariance_matrix.shape[0], "Mismatched dimensions!"
    assert lambda_risk >= 0, "Risk aversion must be non-negative."
    
    n_assets = len(expected_returns)
    
    # --- Mean-Variance Optimization ---
    w = cp.Variable(n_assets)
    constraints = [cp.sum(w) == 1]
    if not allow_shorting:
        constraints.append(w >= 0)
    
    # Objective: Maximize (w^T μ - λ w^T Σ w)
    objective = cp.Maximize(w @ expected_returns - lambda_risk * cp.quad_form(w, covariance_matrix))
    problem = cp.Problem(objective, constraints)
    problem.solve()
    
    optimal_weights = w.value.round(4)
    portfolio_return = (optimal_weights @ expected_returns).round(4)
    portfolio_volatility = np.sqrt(optimal_weights @ covariance_matrix @ optimal_weights).round(4)
    
    # --- Monte Carlo Simulation ---
    np.random.seed(42)  # For reproducibility
    mc_returns = []
    mc_volatilities = []
    mc_weights = []
    
    for _ in tqdm(range(n_simulations), desc="Running Monte Carlo"):
        # Generate random weights
        random_weights = np.random.rand(n_assets)
        random_weights /= random_weights.sum()  # Normalize to sum=1
        if not allow_shorting:
            random_weights = np.abs(random_weights) / np.abs(random_weights).sum()
        
        # Calculate return and volatility
        ret = random_weights @ expected_returns
        vol = np.sqrt(random_weights @ covariance_matrix @ random_weights)
        
        mc_weights.append(random_weights)
        mc_returns.append(ret)
        mc_volatilities.append(vol)
    
    # --- Plotting ---
    plt.figure(figsize=(12, 6))
    
    if plot_monte_carlo:
        plt.scatter(mc_volatilities, mc_returns, c='gray', alpha=0.3, label="Monte Carlo Portfolios")
    
    if plot_frontier:
        # Efficient Frontier Calculation
        frontier_returns = np.linspace(np.min(expected_returns), np.max(expected_returns), 50)
        frontier_volatilities = []
        
        for target_return in frontier_returns:
            w_frontier = cp.Variable(n_assets)
            prob = cp.Problem(
                cp.Minimize(cp.quad_form(w_frontier, covariance_matrix)),
                [cp.sum(w_frontier) == 1, 
                 w_frontier @ expected_returns >= target_return] + 
                ([w_frontier >= 0] if not allow_shorting else [])
            )
            prob.solve()
            frontier_volatilities.append(np.sqrt(prob.value))
        
        plt.plot(frontier_volatilities, frontier_returns, 'b-', linewidth=2, label="Efficient Frontier")
    
    plt.scatter(portfolio_volatility, portfolio_return, c='red', s=200, marker='*', label="Optimal Portfolio")
    plt.xlabel("Volatility (σ)")
    plt.ylabel("Expected Return (μ)")
    plt.title(f"Portfolio Optimization: MVO + Monte Carlo of {n_simulations} Simulated Portfolios")
    plt.legend()
    plt.grid()
    plt.show()
    
    return {
        "optimal_weights": optimal_weights,
        "expected_return": portfolio_return,
        "volatility": portfolio_volatility,
        "monte_carlo_returns": np.array(mc_returns),
        "monte_carlo_volatilities": np.array(mc_volatilities),
        "monte_carlo_weights": np.array(mc_weights)
    }



# Example: 4 assets (Stocks, Bonds, Gold, Crypto)
expected_returns = np.array([0.12, 0.06, 0.04, 0.20])  # μ = [12%, 6%, 4%, 20%]
covariance_matrix = np.array([
    [0.04, 0.001, -0.002, 0.005],   # Stocks
    [0.001, 0.01, 0.003, -0.001],    # Bonds
    [-0.002, 0.003, 0.02, 0.000],    # Gold
    [0.005, -0.001, 0.000, 0.25]     # Crypto (high risk)
])

# Run the optimizer
results = portfolio_optimizer(
    expected_returns, 
    covariance_matrix, 
    lambda_risk=0.5, 
    n_simulations=10000,
    allow_shorting=False,  # Set to True for short-selling
    plot_frontier=True,
    plot_monte_carlo=True
)

print("Optimal Weights:", results["optimal_weights"])
print("Expected Return:", results["expected_return"])
print("Volatility:", results["volatility"])