import matplotlib.pyplot as plt

# Define Market Timing Dashboard Data
indicators = [
    {"category": "Intermarket Ratios", "indicator": "QQQ:SPY", "current_value": 1.25, "signal": "Tech Strength", "color": "green"},
    {"category": "Intermarket Ratios", "indicator": "XLY:XLP", "current_value": 1.18, "signal": "Risk On", "color": "green"},
    {"category": "Intermarket Ratios", "indicator": "IWF:IWD", "current_value": 1.32, "signal": "Growth Leading", "color": "green"},
    {"category": "Intermarket Ratios", "indicator": "TRAN:UTIL", "current_value": 2.45, "signal": "Economic Strength", "color": "green"},

    {"category": "Sentiment Indicators", "indicator": "VIX", "current_value": 16.5, "signal": "Normal", "color": "yellow"},
    {"category": "Sentiment Indicators", "indicator": "Put/Call Ratio", "current_value": 0.82, "signal": "Neutral", "color": "yellow"},

    {"category": "Market Breadth", "indicator": "% Above 50-day MA", "current_value": 68, "signal": "Healthy", "color": "green"},
    {"category": "Market Breadth", "indicator": "% Above 200-day MA", "current_value": 72, "signal": "Bullish", "color": "green"},
    {"category": "Market Breadth", "indicator": "Advance/Decline", "current_value": 1250, "signal": "Positive", "color": "green"},
    {"category": "Market Breadth", "indicator": "New Hi/Lo Ratio", "current_value": 3.2, "signal": "Strong", "color": "green"},
]

# Create Figure
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('off')

# Define Sections and Positions
sections = ["Intermarket Ratios", "Sentiment Indicators", "Market Breadth"]
y_positions = [0.9, 0.5, 0.1]
section_height = 0.25

# Plot Sections
for sec, y_pos in zip(sections, y_positions):
    ax.text(0.01, y_pos + 0.15, sec, fontsize=14, fontweight='bold', color='black')

    # Get indicators in section
    section_indicators = [i for i in indicators if i['category'] == sec]
    for idx, ind in enumerate(section_indicators):
        x = 0.05 + idx * 0.22
        # Box background color by signal color
        rect = plt.Rectangle((x, y_pos), 0.20, 0.10, color=ind['color'], alpha=0.3)
        ax.add_patch(rect)
        ax.text(x + 0.01, y_pos + 0.06, ind['indicator'], fontsize=12, fontweight='semibold')
        ax.text(x + 0.01, y_pos + 0.02, f"Value: {ind['current_value']}", fontsize=11)
        ax.text(x + 0.01, y_pos - 0.02, f"Signal: {ind['signal']}", fontsize=11, style='italic')

plt.tight_layout()
plt.show()

