import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from scipy.optimize import minimize
from fredapi import Fred
import matplotlib.pyplot as plt

# Streamlit setup
st.title("Portfolio Optimizer")

# Sidebar inputs
st.sidebar.header("Input Parameters")
tickers = st.sidebar.text_input("Enter tickers (comma-separated):", "SPY, BND, GLD, QQQ, VTI")
tickers = [ticker.strip() for ticker in tickers.split(',')]
start_date = st.sidebar.date_input("Start Date", datetime.today() - timedelta(5 * 365))
end_date = st.sidebar.date_input("End Date", datetime.today())


# Run optimization when button is clicked
if st.sidebar.button("Optimize Portfolio"):
    with st.spinner('Optimizing portfolio...'):
        try:
            # Download adjusted close price (which include dividends and stock splits)
            adj_close_df = pd.DataFrame()
            for ticker in tickers:
                data = yf.download(ticker, start=start_date, end=end_date)
                adj_close_df[ticker] = data['Adj Close']

            # Calculate the log return
            log_returns = np.log(adj_close_df / adj_close_df.shift(1)).dropna()

            # Calculate the covariance matrix using annualized log returns
            cov_matrix = log_returns.cov() * 252

            # Define the portfolio performance metrics
            def standard_deviation(weights, cov_matrix):
                return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

            def expected_return(weights, log_returns):
                return np.sum(log_returns.mean() * weights) * 252

            def sharpe_ratio(weights, log_returns, risk_free_rate, cov_matrix):
                return (expected_return(weights, log_returns) - risk_free_rate) / standard_deviation(weights, cov_matrix)

            # Define the risk free rate
            #fred = Fred(api_key="")
            #ten_year_treasury_rate = fred.get_series_latest_release('GS10') / 100
            #risk_free_rate = ten_year_treasury_rate.iloc[-1]
            risk_free_rate = 0.05

            # Define the function to minimize the negative Sharpe ratio
            def negative_sharpe_ratio(weights, log_returns, risk_free_rate, cov_matrix):
                return -sharpe_ratio(weights, log_returns, risk_free_rate, cov_matrix)

            # Set the constraints and bounds
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            bounds = [(0, 0.5) for _ in range(len(tickers))]

            # Initial weights
            initial_weights = np.array([1 / len(tickers)] * len(tickers))

            # Optimize the portfolio
            result = minimize(negative_sharpe_ratio, initial_weights, args=(log_returns, risk_free_rate, cov_matrix),
                              method='SLSQP', bounds=bounds, constraints=constraints)
            if not result.success:
                st.error("Optimization failed. Please try again.")
                st.stop()

            optimal_weights = result.x

            optimal_portfolio_return = expected_return(optimal_weights, log_returns)
            optimal_portfolio_volatility = standard_deviation(optimal_weights, cov_matrix)
            optimal_portfolio_sharpe_ratio = sharpe_ratio(optimal_weights, log_returns, risk_free_rate, cov_matrix)

            # Calculate portfolio returns
            portfolio_returns = log_returns.dot(optimal_weights)

            # Calculate VaR at 95% confidence level
            VaR_95 = np.percentile(portfolio_returns, 5)

            # Calculate CVaR at 95% confidence level
            CVaR_95 = portfolio_returns[portfolio_returns <= VaR_95].mean()

            # Plot the optimal weights
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(tickers, optimal_weights, color='skyblue')
            ax.set_xlabel('Assets')
            ax.set_ylabel('Optimal Weights')
            ax.set_title('Optimal Portfolio Weights')
            ax.grid(True)

            # Display the results
            st.pyplot(fig)

            # Additional plot for portfolio returns distribution
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(portfolio_returns, bins=50, color='skyblue', edgecolor='black')
            ax.set_xlabel('Returns')
            ax.set_ylabel('Frequency')
            ax.set_title('Portfolio Returns Distribution')
            ax.axvline(VaR_95, color='r', linestyle='dashed', linewidth=2, label=f'VaR 95%: {VaR_95:.4f}')
            ax.axvline(CVaR_95, color='g', linestyle='dashed', linewidth=2, label=f'CVaR 95%: {CVaR_95:.4f}')
            ax.legend()

            st.pyplot(fig)

            st.write("### Optimal Weights")
            for ticker, weight in zip(tickers, optimal_weights):
                st.write(f"{ticker}: {weight:.4f}")

            st.write("### Metrics")
            st.write(f"Expected Yearly Return: {optimal_portfolio_return:.4f}")
            st.write(f"Standard Deviation: {optimal_portfolio_volatility:.4f}")
            st.write(f"Sharpe Ratio: {optimal_portfolio_sharpe_ratio:.4f}")
            st.write(f"Risk Free Rate (10-Year Treasury Rate): {risk_free_rate * 100:.2f}%")
            st.write(f"Value at Risk (VaR 95%): {VaR_95:.4f}")
            st.write(f"Conditional Value at Risk (CVaR 95%): {CVaR_95:.4f}")

            st.success('Optimization complete!')

        except Exception as e:
            st.error(f"An error occurred: {e}")

st.sidebar.markdown("---")

st.sidebar.link_button("Black-Scholes Calculator", "https://black-scholes.streamlit.app/", help="Opens in new tab")


#re-checking if push is working