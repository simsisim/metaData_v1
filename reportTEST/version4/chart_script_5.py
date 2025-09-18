import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# Create dataframe from the provided data
data = [
    {"ticker": "NVDA", "price": 174.18, "rsi": 35.75, "yearly_return": 38.71},
    {"ticker": "MSFT", "price": 506.69, "rsi": 36.29, "yearly_return": 24.12},
    {"ticker": "AAPL", "price": 232.14, "rsi": 60.39, "yearly_return": 2.97},
    {"ticker": "GOOGL", "price": 212.91, "rsi": 72.45, "yearly_return": 31.37},
    {"ticker": "GOOG", "price": 213.53, "rsi": 72.87, "yearly_return": 30.43},
    {"ticker": "AMZN", "price": 229.00, "rsi": 61.40, "yearly_return": 34.07},
    {"ticker": "META", "price": 738.70, "rsi": 39.23, "yearly_return": 43.43},
    {"ticker": "AVGO", "price": 297.39, "rsi": 45.12, "yearly_return": 90.15},
    {"ticker": "TSLA", "price": 333.87, "rsi": 46.82, "yearly_return": 62.27},
    {"ticker": "JPM", "price": 301.42, "rsi": 69.90, "yearly_return": 39.27},
    {"ticker": "WMT", "price": 96.98, "rsi": 25.53, "yearly_return": 28.71},
    {"ticker": "LLY", "price": 732.58, "rsi": 83.83, "yearly_return": -22.21},
    {"ticker": "V", "price": 351.78, "rsi": 79.30, "yearly_return": 31.60},
    {"ticker": "ORCL", "price": 226.13, "rsi": 27.19, "yearly_return": 65.72},
    {"ticker": "MA", "price": 595.29, "rsi": 74.49, "yearly_return": 26.90},
    {"ticker": "NFLX", "price": 1208.25, "rsi": 46.93, "yearly_return": 76.69},
    {"ticker": "XOM", "price": 114.29, "rsi": 95.49, "yearly_return": 1.63},
    {"ticker": "JNJ", "price": 177.17, "rsi": 67.59, "yearly_return": 11.58},
    {"ticker": "COST", "price": 943.32, "rsi": 29.70, "yearly_return": 6.77},
    {"ticker": "HD", "price": 406.77, "rsi": 62.55, "yearly_return": 11.78}
]

df = pd.DataFrame(data)

# Create color categories based on yearly returns
def get_color_category(return_val):
    if return_val > 50:
        return 'High (>50%)'
    elif return_val >= 0:
        return 'Mid (0-50%)'
    else:
        return 'Low (<0%)'

df['return_cat'] = df['yearly_return'].apply(get_color_category)

# Calculate bubble sizes based on magnitude of yearly returns
df['bubble_size'] = abs(df['yearly_return']) * 0.5 + 5  # Scale for visibility

# Define colors
color_map = {
    'High (>50%)': '#2E8B57',  # Sea green
    'Mid (0-50%)': '#1FB8CD',  # Strong cyan  
    'Low (<0%)': '#DB4545'     # Bright red
}

# Create figure
fig = go.Figure()

# Add scatter points for each category
for category in df['return_cat'].unique():
    category_data = df[df['return_cat'] == category]
    fig.add_trace(go.Scatter(
        x=category_data['rsi'],
        y=category_data['price'],
        mode='markers+text',
        marker=dict(
            size=category_data['bubble_size'],
            color=color_map[category],
            opacity=0.7,
            line=dict(width=1, color='white')
        ),
        text=category_data['ticker'],
        textposition='middle center',
        textfont=dict(size=10, color='white'),
        name=category,
        hovertemplate='<b>%{text}</b><br>' +
                      'RSI: %{x:.1f}<br>' +
                      'Price: $%{y:.2f}<br>' +
                      'Yearly Return: %{customdata:.1f}%<extra></extra>',
        customdata=category_data['yearly_return']
    ))

# Add vertical lines at RSI 30 and 70
fig.add_vline(x=30, line_dash="dash", line_color="gray", opacity=0.7)
fig.add_vline(x=70, line_dash="dash", line_color="gray", opacity=0.7)

# Update layout
fig.update_layout(
    title='Price vs RSI by Return',
    xaxis_title='RSI',
    yaxis_title='Price ($)',
    legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5)
)

# Update traces
fig.update_traces(cliponaxis=False)

# Update axes
fig.update_xaxes(range=[20, 100])
fig.update_yaxes(tickformat='$,.0f')

# Save the chart
fig.write_image('stock_rsi_scatter.png')