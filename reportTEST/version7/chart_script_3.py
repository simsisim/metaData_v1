import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Create DataFrame from the provided data including RSI and momentum
data = {
    "ticker": ["APP","PLTR","GEV","AVGO","NFLX","ORCL","TSLA","C","GE","ANET","NTRS","VRSN","META","CSCO","PM","UNH","LLY","LULU","ACN","ASML","HPQ","AMGN","XLV","AZN","PG"],
    "daily_year_pct_change": [443.85,416.17,233.79,90.15,76.69,65.72,62.27,61.64,61.15,60.1,52.62,51.72,43.43,42.4,41.71,-46.36,-22.21,-21.88,-21.53,-15.77,-14.7,-10.49,-10.28,-6.59,-4.76],
    "daily_rsi_14": [54.17,24.14,36.01,45.12,46.93,27.19,46.82,64.83,50.0,47.71,68.08,59.95,39.23,42.39,44.2,81.05,83.83,64.24,79.57,57.62,73.86,53.73,75.72,82.07,56.82],
    "daily_momentum_20": [14.02,-10.54,-5.34,-0.17,3.7,-7.29,8.67,4.61,-0.19,2.69,4.24,0.18,-2.56,-1.27,-0.16,23.35,2.54,5.08,4.08,7.92,9.5,-1.39,4.41,8.35,2.71],
    "Performance_Category": ["Top Performer","Top Performer","Top Performer","Top Performer","Top Performer","Top Performer","Top Performer","Top Performer","Top Performer","Top Performer","Top Performer","Top Performer","Top Performer","Top Performer","Top Performer","Bottom Performer","Bottom Performer","Bottom Performer","Bottom Performer","Bottom Performer","Bottom Performer","Bottom Performer","Bottom Performer","Bottom Performer","Bottom Performer"]
}

df = pd.DataFrame(data)

# Create color mapping
color_map = {
    'Top Performer': '#2E8B57',  # Sea green
    'Bottom Performer': '#DB4545'  # Bright red
}

# Create scatter plot showing RSI vs Momentum
fig = go.Figure()

for category in ['Top Performer', 'Bottom Performer']:
    subset = df[df['Performance_Category'] == category]
    
    fig.add_trace(go.Scatter(
        x=subset['daily_rsi_14'],
        y=subset['daily_momentum_20'],
        mode='markers+text',
        name=category.replace(' Performer', ''),
        marker=dict(
            color=color_map[category],
            size=12,
            line=dict(width=1, color='white')
        ),
        text=subset['ticker'],
        textposition='top center',
        hovertemplate='<b>%{text}</b><br>' +
                      'RSI: %{x:.1f}<br>' +
                      'Momentum: %{y:.1f}<br>' +
                      'Annual Return: ' + subset['daily_year_pct_change'].astype(str) + '%<br>' +
                      '<extra></extra>'
    ))

# Update layout
fig.update_layout(
    title='RSI vs Momentum Analysis',
    xaxis_title='RSI (14-day)',
    yaxis_title='Momentum (20-day)',
    showlegend=True,
    legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5)
)

# Update traces
fig.update_traces(cliponaxis=False)

# Update axes formatting
fig.update_xaxes(range=[0, 100])
fig.update_yaxes(tickformat='.1f')

# Save the chart
fig.write_image('rsi_momentum_scatter.png')