import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# Data
data = [
    {"ticker": "NVDA", "atr_pct": 2.84, "yearly_return": 38.71, "price": 174.18, "rsi": 35.75},
    {"ticker": "MSFT", "atr_pct": 1.52, "yearly_return": 24.12, "price": 506.69, "rsi": 36.29},
    {"ticker": "AAPL", "atr_pct": 1.84, "yearly_return": 2.97, "price": 232.14, "rsi": 60.39},
    {"ticker": "GOOGL", "atr_pct": 2.07, "yearly_return": 31.37, "price": 212.91, "rsi": 72.45},
    {"ticker": "GOOG", "atr_pct": 2.04, "yearly_return": 30.43, "price": 213.53, "rsi": 72.87},
    {"ticker": "AMZN", "atr_pct": 2.00, "yearly_return": 34.07, "price": 229.00, "rsi": 61.40},
    {"ticker": "META", "atr_pct": 2.27, "yearly_return": 43.43, "price": 738.70, "rsi": 39.23},
    {"ticker": "AVGO", "atr_pct": 3.18, "yearly_return": 90.15, "price": 297.39, "rsi": 45.12},
    {"ticker": "TSLA", "atr_pct": 3.76, "yearly_return": 62.27, "price": 333.87, "rsi": 46.82},
    {"ticker": "JPM", "atr_pct": 1.56, "yearly_return": 39.27, "price": 301.42, "rsi": 69.90},
    {"ticker": "WMT", "atr_pct": 1.67, "yearly_return": 28.71, "price": 96.98, "rsi": 25.53},
    {"ticker": "LLY", "atr_pct": 2.99, "yearly_return": -22.21, "price": 732.58, "rsi": 83.83},
    {"ticker": "V", "atr_pct": 1.47, "yearly_return": 31.60, "price": 351.78, "rsi": 79.30},
    {"ticker": "ORCL", "atr_pct": 3.39, "yearly_return": 65.72, "price": 226.13, "rsi": 27.19},
    {"ticker": "MA", "atr_pct": 1.37, "yearly_return": 26.90, "price": 595.29, "rsi": 74.49}
]

# Extract data for plotting
tickers = [d["ticker"] for d in data]
atr_pcts = [d["atr_pct"] for d in data]
yearly_returns = [d["yearly_return"] for d in data]
prices = [d["price"] for d in data]
rsi_values = [d["rsi"] for d in data]

# Categorize by RSI
colors = []
for rsi in rsi_values:
    if rsi > 70:
        colors.append("#DB4545")  # Red for overbought
    elif rsi < 30:
        colors.append("#2E8B57")  # Green for oversold
    else:
        colors.append("#1FB8CD")  # Blue for neutral

# Scale bubble sizes (normalize prices for better visualization)
min_price, max_price = min(prices), max(prices)
min_size, max_size = 10, 50
bubble_sizes = [min_size + (price - min_price) / (max_price - min_price) * (max_size - min_size) for price in prices]

# Calculate median ATR
median_atr = np.median(atr_pcts)

# Create scatter plot
fig = go.Figure()

# Add scatter points
fig.add_trace(go.Scatter(
    x=atr_pcts,
    y=yearly_returns,
    mode='markers+text',
    marker=dict(
        size=bubble_sizes,
        color=colors,
        opacity=0.7,
        line=dict(width=1, color='white')
    ),
    text=tickers,
    textposition='top center',
    textfont=dict(size=10),
    hovertemplate='<b>%{text}</b><br>' +
                  'ATR: %{x:.2f}%<br>' +
                  'Return: %{y:.1f}%<br>' +
                  'Price: $%{customdata}<br>' +
                  '<extra></extra>',
    customdata=prices,
    showlegend=False
))

# Add quadrant lines
# Horizontal line at 0% return
fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

# Vertical line at median ATR
fig.add_vline(x=median_atr, line_dash="dash", line_color="gray", opacity=0.5)

# Update layout
fig.update_layout(
    title="Risk-Return Analysis",
    xaxis_title="ATR %",
    yaxis_title="Yearly Return %",
    showlegend=False
)

# Update traces
fig.update_traces(cliponaxis=False)

# Save the chart
fig.write_image("risk_return_analysis.png")