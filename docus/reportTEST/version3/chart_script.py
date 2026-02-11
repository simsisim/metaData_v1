import plotly.graph_objects as go
import pandas as pd

# Data from the provided JSON
data = [
    {"Industry": "Pharmaceuticals: major", "Count": 3},
    {"Industry": "Major banks", "Count": 3},
    {"Industry": "Finance/Rental/Leasing", "Count": 2},
    {"Industry": "Internet software/Services", "Count": 2},
    {"Industry": "Telecommunications equipment", "Count": 1},
    {"Industry": "Computer peripherals", "Count": 1},
    {"Industry": "Integrated oil", "Count": 1},
    {"Industry": "Internet retail", "Count": 1},
    {"Industry": "Aerospace & defense", "Count": 1},
    {"Industry": "Electric utilities", "Count": 1},
    {"Industry": "Home improvement chains", "Count": 1},
    {"Industry": "Information technology services", "Count": 1},
    {"Industry": "Investment managers", "Count": 1},
    {"Industry": "Semiconductors", "Count": 1},
    {"Industry": "Wireless telecommunications", "Count": 1},
    {"Industry": "Managed health care", "Count": 1}
]

# Convert to DataFrame
df = pd.DataFrame(data)

# Sort by count in descending order
df = df.sort_values('Count', ascending=True)  # ascending=True for horizontal bars to show highest at top

# Abbreviate industry names to 15 characters or less
industry_abbreviations = {
    "Pharmaceuticals: major": "Pharma: major",
    "Major banks": "Major banks",
    "Finance/Rental/Leasing": "Finance/Rental",
    "Internet software/Services": "Internet sw/svc",
    "Telecommunications equipment": "Telecom equip",
    "Computer peripherals": "Computer periph",
    "Integrated oil": "Integrated oil",
    "Internet retail": "Internet retail",
    "Aerospace & defense": "Aerospace & def",
    "Electric utilities": "Electric utils",
    "Home improvement chains": "Home improv",
    "Information technology services": "IT services",
    "Investment managers": "Investment mgrs",
    "Semiconductors": "Semiconductors",
    "Wireless telecommunications": "Wireless tel",
    "Managed health care": "Managed health"
}

df['Industry_Short'] = df['Industry'].map(industry_abbreviations)

# Brand colors in order
colors = ['#1FB8CD', '#DB4545', '#2E8B57', '#5D878F', '#D2BA4C', '#B4413C', '#964325', '#944454', '#13343B', '#DB4545']

# Create a color list that cycles through the brand colors
industry_colors = [colors[i % len(colors)] for i in range(len(df))]

# Create horizontal bar chart
fig = go.Figure(data=[
    go.Bar(
        y=df['Industry_Short'],
        x=df['Count'],
        orientation='h',
        marker_color=industry_colors,
        text=df['Count'],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Count: %{x}<extra></extra>'
    )
])

# Update layout
fig.update_layout(
    title='Bullish Trend Stocks by Industry',
    xaxis_title='Stock Count',
    yaxis_title='Industry',
    showlegend=False
)

# Update traces
fig.update_traces(cliponaxis=False)

# Save the chart
fig.write_image('bullish_stocks_by_industry.png')