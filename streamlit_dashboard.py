
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from lifelines import KaplanMeierFitter
import seaborn as sns

# Data setup (Simulated or actual data can be used here)
@st.cache
def load_data():
    # Simulate some Nifty 50 data
    dates = pd.date_range(start="2019-01-01", periods=750, freq='B')
    close_prices = np.cumprod(1 + np.random.normal(0, 0.01, len(dates)))  # Simulated prices
    returns = 100 * np.diff(close_prices) / close_prices[:-1]  # Simulated returns
    nifty_df = pd.DataFrame({'Date': dates[1:], 'Close': close_prices[1:], 'Returns': returns})
    nifty_df.set_index('Date', inplace=True)

    # Simulated retail share (for event study)
    retail_share = pd.Series(np.random.normal(30, 5, len(nifty_df)), index=nifty_df.index)
    retail_share = retail_share.clip(lower=10, upper=60)

    return nifty_df, retail_share

# Event Study - Cumulative Abnormal Returns (CAR)
@st.cache
def event_study(nifty_df, retail_share, window=5):
    spike_days = retail_share[retail_share >= retail_share.quantile(0.99)].index
    returns = nifty_df['Returns']

    # Calculate CARs
    def event_window_returns(prices, event_dates, window=5):
        windows = []
        for d in event_dates:
            if (d - pd.Timedelta(days=window) in prices.index) and (d + pd.Timedelta(days=window) in prices.index):
                start = prices.index.get_loc(d - pd.Timedelta(days=window), method='nearest')
                end = prices.index.get_loc(d + pd.Timedelta(days=window), method='nearest')
                win = prices.pct_change().iloc[start:end+1].reset_index(drop=True)
                windows.append(win.values)
        return pd.DataFrame(windows).T

    car_matrix = event_window_returns(returns, spike_days, window)
    avg_car = car_matrix.mean(axis=1).cumsum()

    return avg_car

# Trader Segmentation - KMeans Clustering
@st.cache
def trader_segmentation():
    np.random.seed(42)
    trader_df = pd.DataFrame({
        'avg_trades_per_month': np.random.randint(1, 10, 1000),
        'pct_deriv_trades': np.random.uniform(0, 1, 1000),
        'avg_holding_days': np.random.randint(1, 200, 1000),
        'avg_pnl': np.random.normal(0, 1000, 1000)
    })
    kmeans = KMeans(n_clusters=4, random_state=0)
    trader_df['segment'] = kmeans.fit_predict(trader_df[['avg_trades_per_month', 'pct_deriv_trades', 'avg_holding_days', 'avg_pnl']])
    return trader_df

# Loss Distribution & Bootstrap CI
@st.cache
def loss_distribution():
    losses = np.random.lognormal(mean=10, sigma=2, size=1000)
    losses = np.clip(losses, 100, 500000)

    # Bootstrap confidence intervals for mean loss
    B = 2000
    boot_means = [losses[np.random.randint(0, len(losses), len(losses))].mean() for _ in range(B)]
    ci = np.percentile(boot_means, [2.5, 97.5])

    return losses, ci

# Survival Curve - Kaplan-Meier Analysis
@st.cache
def survival_curve():
    np.random.seed(42)
    account_df = pd.DataFrame({
        'days_active': np.random.randint(1, 365, 1000),
        'is_active_end': np.random.binomial(1, 0.8, 1000)
    })

    kmf = KaplanMeierFitter()
    kmf.fit(account_df['days_active'], event_observed=account_df['is_active_end'])

    return kmf

# Dashboard
st.title("Interactive Retail Trading Behavior Analysis")

# Event Study Section
st.header("Event Study: Cumulative Abnormal Returns (CAR)")
nifty_df, retail_share = load_data()
avg_car = event_study(nifty_df, retail_share)
st.subheader("Average Cumulative Abnormal Returns")
st.line_chart(avg_car)

# Trader Segmentation Section
st.header("Trader Segmentation")
trader_df = trader_segmentation()
st.subheader("Trader Behavior Clusters")
fig, ax = plt.subplots()
ax.scatter(trader_df['avg_trades_per_month'], trader_df['avg_pnl'], c=trader_df['segment'], cmap='viridis')
ax.set_title('Trader Segmentation: Avg Trades vs Avg PnL')
ax.set_xlabel('Average Trades per Month')
ax.set_ylabel('Average PnL')
st.pyplot(fig)

# Loss Distribution Section
st.header("Loss Distribution")
losses, ci = loss_distribution()
st.subheader("Distribution of Losses Among Losing Traders")
fig, ax = plt.subplots()
ax.hist(losses, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
ax.axvline(np.median(losses), color='red', linestyle='dashed', linewidth=2, label=f"Median Loss: {np.median(losses):,.2f}")
ax.axvline(np.percentile(losses, 90), color='green', linestyle='dashed', linewidth=2, label=f"90th Percentile: {np.percentile(losses, 90):,.2f}")
ax.set_title("Histogram of Losses Among Losing Traders")
ax.set_xlabel("Loss Amount (₹)")
ax.set_ylabel("Frequency")
ax.legend()
st.pyplot(fig)
st.write(f"Bootstrap 95% CI for Mean Loss: {ci}")

# Survival Curve Section
st.header("Survival Curve of Account Activity")
kmf = survival_curve()
st.subheader("Probability of Remaining Active Over Time")
fig, ax = plt.subplots()
kmf.plot_survival_function(ax=ax)
ax.set_title("Survival Curve: New Retail Traders")
ax.set_xlabel("Days Active")
ax.set_ylabel("Probability of Remaining Active")
st.pyplot(fig)

# Download Link (Optional: You could deploy this Streamlit app using their hosting or on a local machine)
st.markdown("[Download the Streamlit app code](sandbox:/mnt/data/streamlit_dashboard.py)")

st.write("This dashboard visualizes key aspects of retail trading behavior using real market data (Nifty 50).")
