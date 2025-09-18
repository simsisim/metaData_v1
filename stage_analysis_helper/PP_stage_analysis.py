import pandas as pd
import plotly.express as px
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Spacer, Paragraph, Image as RLImage
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

# ---- File paths ----
input_file = "results/stage_analysis/stage_analysis_0-5_daily_20250905.csv"
pdf_file = "stage_analysis_stages_report.pdf"
pie_chart_file = "market_stage_pie_chart.png"

# ---- Read CSV ----
df = pd.read_csv(input_file)

# ---- Prepare data for pie chart ----
stage_counts = df['daily_sa_name'].value_counts().reset_index()
stage_counts.columns = ['stage', 'count']

# Map your color codes here
color_map = {
    'Bullish Trend': '#388e3c',
    'Bullish Fade': '#FF9800',
    'Bearish Trend': '#F44336',
    'Pullback': '#D4C464',
    'Mean Reversion': '#C0AF53',
    'Launch Pad': '#C2A7D4',
    # Add more stage colors as needed
}

# --- Plotly Pie Chart ---
fig = px.pie(stage_counts, 
             values='count', 
             names='stage',
             title='Market Stage Analysis Distribution',
             color='stage',
             color_discrete_map=color_map)
fig.update_traces(
    textposition='inside',
    textinfo='label+percent+value',
    hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
)
fig.update_layout(
    uniformtext_minsize=14, 
    uniformtext_mode='hide'
)
fig.write_image(pie_chart_file)

# ---- Get unique stages and color codes for table ----
stage_triplets = df[['daily_sa_name', 'daily_sa_code', 'daily_sa_color_code']].drop_duplicates().values.tolist()

columns = []
for sa_name, sa_code, sa_color in stage_triplets:
    tickers = df[df['daily_sa_name'] == sa_name]['ticker'].tolist()
    col = [sa_name, sa_code] + tickers
    columns.append((col, sa_color))

max_len = max(len(col) for col, _ in columns)
table_data = []
for col, _ in columns:
    col += [''] * (max_len - len(col))
    table_data.append(col)

table_data = list(map(list, zip(*[col for col, _ in columns])))
num_stages = len(columns)

CODE_TO_COLOR = {
    'green_light': colors.HexColor('#C8E6C9'),
    'orange_light': colors.HexColor('#FFE0B2'),
    'purple': colors.HexColor('#E1BEE7'),
    # Extend as needed for more stage colors
}
col_bg_colors = [CODE_TO_COLOR.get(color_code, colors.white) for _, color_code in columns]

table_style = TableStyle([
    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
    ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
    ('FONTNAME', (0,0), (-1,1), 'Helvetica-Bold'),
    ('FONTSIZE', (0,0), (-1,-1), 8)
])
for col_idx, bg_color in enumerate(col_bg_colors):
    table_style.add('BACKGROUND', (col_idx, 0), (col_idx, max_len-1), bg_color)

# ---- Build PDF ----
style = getSampleStyleSheet()
doc = SimpleDocTemplate(
    pdf_file,
    pagesize=letter,
    rightMargin=14, leftMargin=14, topMargin=24, bottomMargin=24
)
story = []

story.append(Paragraph("Market Stage Analysis", style['Title']))
story.append(Spacer(1, 14))

# Add pie chart (image)
story.append(Paragraph("Market Stage Analysis Distribution", style['Heading2']))
story.append(RLImage(pie_chart_file, width=350, height=220))
story.append(Spacer(1, 18))

# Add stage analysis table
story.append(Paragraph("Stage Table by Market Stage", style['Heading2']))
story.append(Spacer(1, 12))
table = Table(table_data, colWidths= [60]* num_stages)
table.setStyle(table_style)
story.append(table)

doc.build(story)
print(f"ReportLab PDF with pie chart and table saved as {pdf_file}.")

