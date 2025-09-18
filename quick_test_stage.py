#!/usr/bin/env python3
import sys
import pandas as pd
from pathlib import Path
sys.path.append('src')

try:
    print("Loading CSV...")
    csv_file = 'results/stage_analysis/stage_analysis_2-5_daily_20250905.csv'
    df = pd.read_csv(csv_file)
    print(f"✅ CSV loaded: {len(df)} rows")

    print("Testing stage detection...")
    from report_generators.universal_png_pdf_generator import UniversalPngPdfGenerator
    generator = UniversalPngPdfGenerator()
    chart_type = generator.detect_chart_type(csv_file, df)
    print(f"✅ Chart type detected: {chart_type}")

    print("Testing PNG generation...")
    png_path = generator.generate_png_from_csv(csv_file)
    print(f"✅ PNG generated: {png_path}")

    print("Testing PDF generation...")
    pdf_path = generator.generate_pdf_report(csv_file, png_path)
    print(f"✅ PDF generated: {pdf_path}")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()