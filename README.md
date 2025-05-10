# Portfolio Optimizer with Risk Aversion

This project is a **Streamlit** application for portfolio optimization using **Mean-Variance Optimization (MVO)** and **Monte Carlo simulations**. It allows users to input asset data, customize risk aversion, and visualize the efficient frontier.

## Features
- **Mean-Variance Optimization**: Calculate optimal portfolio weights based on expected returns, covariance matrix, and risk aversion.
- **Monte Carlo Simulations**: Generate random portfolios to explore return and volatility distributions.
- **Efficient Frontier**: Visualize the trade-off between risk and return.
- **Rebalancing Calculator**: Compute recommended allocation amounts based on initial capital.

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd MVO
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Run the Streamlit app:
   ```bash
   streamlit run scripts/app.py
   ```
2. Open the app in your browser at `http://localhost:8501`.

## File Structure
- `scripts/app.py`: Main application file.
- `requirements.txt`: List of Python dependencies.

## Dependencies
- `numpy`
- `pandas`
- `streamlit`
- `matplotlib`
- `cvxpy`

## License
This project is licensed under the MIT License.

## Mathematical Explanation of Mean-Variance Optimization (MVO)

Mean-Variance Optimization (MVO) is a mathematical framework for constructing an investment portfolio that maximizes expected return for a given level of risk or minimizes risk for a given level of expected return. It was introduced by Harry Markowitz in 1952 and forms the foundation of modern portfolio theory.

### Objective Function
The MVO problem can be formulated as:

$$
\max_{\mathbf{w}} \mathbf{w}^T \mathbf{\mu} - \lambda \mathbf{w}^T \Sigma \mathbf{w}
$$

Where:
- $\mathbf{w}$: Vector of portfolio weights (decision variables).
- $\mu$: Vector of expected returns for each asset.
- $\Sigma$: Covariance matrix of asset returns.
- $\lambda$: Risk aversion parameter (higher \(\lambda\) implies more conservative portfolios).

### Constraints
1. **Budget Constraint**: The sum of portfolio weights must equal 1:
   $$
   \sum_{i=1}^n w_i = 1
   $$
2. **Non-Negativity Constraint** (if short-selling is not allowed):
   $$
   w_i \geq 0 \quad \forall i
   $$

### Solution
The optimization problem is solved using quadratic programming, as the objective function is quadratic and the constraints are linear.

### Limitations
1. **Sensitivity to Input Estimates**: MVO is highly sensitive to the accuracy of expected returns and covariance matrix estimates. Small errors can lead to significantly different portfolio allocations.
2. **Single-Period Model**: Assumes a single investment period, which may not reflect real-world scenarios.
3. **No Transaction Costs**: Ignores transaction costs and taxes.
4. **Normality Assumption**: Assumes asset returns are normally distributed, which may not hold in practice.

### Assumptions
1. Investors are rational and risk-averse.
2. Asset returns are jointly normally distributed.
3. Markets are frictionless (no transaction costs or taxes).
4. Portfolio risk is measured solely by variance.

### Usage
MVO is widely used in portfolio management to:
- Construct efficient portfolios that balance risk and return.
- Visualize the efficient frontier, which represents the set of optimal portfolios.
- Analyze the impact of risk aversion on portfolio allocation.

In this application, MVO is implemented using the `cvxpy` library for convex optimization. Users can customize risk aversion, allow or disallow short-selling, and input their own asset data to compute optimal portfolio weights.

## Monte Carlo Simulations

Monte Carlo simulations are a statistical technique used to model and analyze the behavior of complex systems by generating random samples. In the context of portfolio optimization, Monte Carlo simulations are used to explore the distribution of portfolio returns and volatilities by randomly generating portfolio weights.

### Methodology
1. **Random Weight Generation**:
   - If short-selling is allowed, weights are sampled from a normal distribution and normalized to sum to 1.
   - If short-selling is not allowed, weights are sampled from a Dirichlet distribution to ensure non-negativity and sum-to-one constraints.
2. **Portfolio Return and Volatility**:
   - For each set of weights $\mathbf{w}$:
     - Portfolio return: $\mathbf{w}^T \mathbf{\mu}$
     - Portfolio volatility: $\sqrt{\mathbf{w}^T \Sigma \mathbf{w}}$
3. **Simulation**:
   - Repeat the process for a large number of iterations (e.g., 5000) to generate a distribution of portfolio returns and volatilities.

### Applications
- **Exploration of Portfolio Space**: Monte Carlo simulations provide insights into the range of possible portfolio outcomes, helping investors understand the trade-offs between risk and return.
- **Validation of Optimization Results**: By comparing the optimal portfolio to randomly generated portfolios, investors can assess the robustness of the optimization results.
- **Visualization**: The scatter plot of simulated portfolios helps visualize the efficient frontier and the position of the optimal portfolio.

### Limitations
1. **Computationally Intensive**: Generating a large number of random portfolios can be computationally expensive.
2. **Randomness**: Results depend on the quality and size of the random samples.
3. **Simplistic Assumptions**: Assumes static input parameters (expected returns and covariance matrix) and does not account for dynamic market conditions.

### Usage in This Application
In this application, Monte Carlo simulations are implemented to:
- Generate random portfolios based on user-defined parameters (expected returns, covariance matrix, and short-selling constraints).
- Visualize the distribution of portfolio returns and volatilities alongside the efficient frontier.
- Provide a comparative context for the optimal portfolio derived from Mean-Variance Optimization.
