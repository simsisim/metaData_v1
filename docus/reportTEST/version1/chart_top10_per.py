import plotly.express as px
import pandas as pd

# Create DataFrame from the provided data
data = [
    {"Ticker": "WMT", "RS_Percentile": 99},
    {"Ticker": "MA", "RS_Percentile": 93},
    {"Ticker": "GOOGL", "RS_Percentile": 87},
    {"Ticker": "V", "RS_Percentile": 81},
    {"Ticker": "GOOG", "RS_Percentile": 74},
    {"Ticker": "JPM", "RS_Percentile": 68},
    {"Ticker": "LLY", "RS_Percentile": 62},
    {"Ticker": "AAPL", "RS_Percentile": 56},
    {"Ticker": "MSFT", "RS_Percentile": 50},
    {"Ticker": "AMZN", "RS_Percentile": 44}
]

df = pd.DataFrame(data)

# Create bar chart with color scale based on RS percentile
fig = px.bar(df, 
             x='Ticker', 
             y='RS_Percentile',
             color='RS_Percentile',
             color_continuous_scale='Viridis',
             title='Top 10 Stocks by 1-Day RS Percentile')

# Update layout and styling
fig.update_layout(
    xaxis_title='Ticker',
    yaxis_title='RS Percentile',
    coloraxis_colorbar_title='RS Percentile'
)

# Update traces to follow guidelines
fig.update_traces(cliponaxis=False)

# Save the chart
fig.write_image('top_10_stocks_rs_percentile.png')