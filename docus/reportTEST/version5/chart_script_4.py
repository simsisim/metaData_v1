import pandas as pd
import plotly.graph_objects as go
import numpy as np

# Load the data
df = pd.read_csv("basic_calculation_daily_0_20250829.csv")

# Define the time period columns and their labels
time_periods = {
    '1D': 'daily_daily_daily_1d_pct_change',
    '3D': 'daily_daily_daily_3d_pct_change', 
    '5D': 'daily_daily_daily_5d_pct_change',
    '7D': 'daily_daily_weekly_7d_pct_change',
    '14D': 'daily_daily_weekly_14d_pct_change',
    '22D': 'daily_daily_monthly_22d_pct_change',
    '44D': 'daily_daily_monthly_44d_pct_change',
    '66D': 'daily_daily_quarterly_66d_pct_change',
    '132D': 'daily_daily_quarterly_132d_pct_change',
    '252D': 'daily_daily_yearly_252d_pct_change'
}

# Calculate statistics for each time period
results = []
for period_label, column in time_periods.items():
    if column in df.columns:
        values = df[column].dropna()
        mean_return = values.mean()
        std_dev = values.std()
        positive_count = (values > 0).sum()
        total_count = len(values)
        positive_pct = (positive_count / total_count * 100) if total_count > 0 else 0
        
        results.append({
            'period': period_label,
            'mean_return': mean_return,
            'std_dev': std_dev,
            'positive_pct': positive_pct
        })

# Create DataFrame from results
performance_df = pd.DataFrame(results)

# Create the chart
fig = go.Figure()

# Add line with error bars for mean returns
fig.add_trace(go.Scatter(
    x=performance_df['period'],
    y=performance_df['mean_return'],
    mode='lines+markers',
    name='Avg Return',
    line=dict(color='#1FB8CD', width=3),
    marker=dict(
        size=8,
        color=performance_df['positive_pct'],
        colorscale='RdYlGn',
        cmin=0,
        cmax=100,
        colorbar=dict(
            title=dict(text='Pos Ret %'),
            x=1.02,
            len=0.7
        ),
        showscale=True
    ),
    error_y=dict(
        type='data',
        array=performance_df['std_dev'],
        visible=True,
        color='#1FB8CD',
        thickness=2
    ),
    hovertemplate='Period: %{x}<br>Avg Ret: %{y:.2f}%<br>Std Dev: %{error_y.array:.2f}%<br>Pos Ret: %{marker.color:.1f}%<extra></extra>'
))

# Update layout
fig.update_layout(
    title='Performance Across Time Periods',
    xaxis_title='Time Period',
    yaxis_title='Avg Return (%)',
    showlegend=False
)

# Update traces
fig.update_traces(cliponaxis=False)

# Save the chart
fig.write_image("performance_chart.png")
fig.show()