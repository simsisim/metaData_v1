import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load and merge the data
df_calc = pd.read_csv('basic_calculation_daily_0_20250829.csv')
df_universe = pd.read_csv('tradingview_universe.csv')

# Merge on ticker/Symbol
df = df_calc.merge(df_universe, left_on='ticker', right_on='Symbol', how='inner')

# Convert market cap to billions for better sizing
df['market_cap_billions'] = df['Market capitalization'] / 1e9

# Filter out any invalid data
df = df.dropna(subset=['atr_pct', 'daily_daily_yearly_252d_pct_change', 'Sector', 'market_cap_billions'])

# Create the scatter plot without text labels (put ticker in hover instead)
fig = px.scatter(
    df, 
    x='atr_pct', 
    y='daily_daily_yearly_252d_pct_change',
    color='Sector',
    size='market_cap_billions',
    hover_data=['ticker', 'market_cap_billions'],
    title='Risk vs Return Analysis',
    color_discrete_sequence=['#1FB8CD', '#DB4545', '#2E8B57', '#5D878F', '#D2BA4C', '#B4413C', '#964325', '#944454', '#13343B']
)

# Update traces for better visibility
fig.update_traces(cliponaxis=False)

# Add quadrant lines
fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
fig.add_vline(x=2, line_dash="dash", line_color="gray", opacity=0.5)

# Update layout with shorter axis labels
fig.update_layout(
    xaxis_title="Volatility %",
    yaxis_title="1-Year Return %"
)

# Since there are more than 5 sectors, keep legend vertical (default)
# Update axes
fig.update_xaxes(showgrid=True)
fig.update_yaxes(showgrid=True)

# Save the chart
fig.write_image("risk_return_scatter.png")
print(f"Chart saved successfully! Data points: {len(df)}")
print(f"Sectors: {len(df['Sector'].unique())} unique sectors")