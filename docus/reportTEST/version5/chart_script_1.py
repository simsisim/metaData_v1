import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy import stats

# Load the top performers data
df = pd.read_csv("top_performers_1year.csv")

# Check number of unique sectors for legend placement
print(f"Number of unique sectors: {df['Sector'].nunique()}")
print("Sectors:", df['Sector'].unique())

# Create the scatter plot
fig = px.scatter(df, 
                 x='market_cap_billions', 
                 y='daily_daily_yearly_252d_pct_change',
                 color='Sector',
                 text='ticker',
                 title='Market Cap vs 1-Year Performance',
                 labels={
                     'market_cap_billions': 'Market Cap (b)', 
                     'daily_daily_yearly_252d_pct_change': '1-Year Perf (%)'
                 })

# Update layout for logarithmic x-axis
fig.update_xaxes(type="log", title="Market Cap (b)")
fig.update_yaxes(title="1-Year Perf (%)")

# Position text labels
fig.update_traces(textposition="middle right", cliponaxis=False)

# Add trend line
# Calculate trend line using log of x values
x_vals = np.log10(df['market_cap_billions'])
y_vals = df['daily_daily_yearly_252d_pct_change']
slope, intercept, r_value, p_value, std_err = stats.linregress(x_vals, y_vals)

# Create trend line points
x_trend = np.logspace(np.log10(df['market_cap_billions'].min()), 
                      np.log10(df['market_cap_billions'].max()), 100)
y_trend = slope * np.log10(x_trend) + intercept

fig.add_trace(go.Scatter(x=x_trend, y=y_trend, 
                        mode='lines', 
                        name='Trend',
                        line=dict(color='gray', dash='dash')))

# Center legend since we have 5 sectors or fewer
if df['Sector'].nunique() <= 5:
    fig.update_layout(legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5))

# Save the chart
fig.write_image("market_cap_performance_scatter.png")
print("Chart saved as market_cap_performance_scatter.png")