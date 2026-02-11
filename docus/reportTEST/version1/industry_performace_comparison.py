import pandas as pd
import plotly.graph_objects as go

# Data
data = [
    {"Industry": "Internet software/Services", "Short_Term_Avg": 89.0, "Long_Term_Avg": 82.3, "ST_LT_Diff": 6.7},
    {"Industry": "Finance/Rental/Leasing", "Short_Term_Avg": 84.0, "Long_Term_Avg": 50.0, "ST_LT_Diff": 34.0},
    {"Industry": "Packaged software", "Short_Term_Avg": 45.2, "Long_Term_Avg": 34.0, "ST_LT_Diff": 11.2},
    {"Industry": "Semiconductors", "Short_Term_Avg": 30.8, "Long_Term_Avg": 82.7, "ST_LT_Diff": -51.9}
]

df = pd.DataFrame(data)

# Shorten industry names to fit 15 character limit
df['Industry_Short'] = df['Industry'].str.replace('Internet software/Services', 'Internet SW/Svc')
df['Industry_Short'] = df['Industry_Short'].str.replace('Finance/Rental/Leasing', 'Finance/Rent')
df['Industry_Short'] = df['Industry_Short'].str.replace('Packaged software', 'Packaged SW')

# Brand colors
colors = ['#1FB8CD', '#DB4545', '#2E8B57']

# Create grouped bar chart
fig = go.Figure()

# Add bars for each metric
fig.add_trace(go.Bar(
    name='Short-term Avg',
    x=df['Industry_Short'],
    y=df['Short_Term_Avg'],
    marker_color=colors[0]
))

fig.add_trace(go.Bar(
    name='Long-term Avg',
    x=df['Industry_Short'],
    y=df['Long_Term_Avg'],
    marker_color=colors[1]
))

fig.add_trace(go.Bar(
    name='ST-LT Diff',
    x=df['Industry_Short'],
    y=df['ST_LT_Diff'],
    marker_color=colors[2]
))

# Update layout
fig.update_layout(
    title='Industry Performance Comparison',
    xaxis_title='Industry',
    yaxis_title='Performance',
    barmode='group',
    legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5)
)

# Update traces
fig.update_traces(cliponaxis=False)

# Save chart
fig.write_image('industry_performance_comparison.png')