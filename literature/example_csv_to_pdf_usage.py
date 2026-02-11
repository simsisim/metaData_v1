#!/usr/bin/env python3
"""
Example usage of the Universal PNG-PDF Generator.
Shows how to integrate CSV → PNG + PDF generation into existing workflows.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from report_generators.universal_png_pdf_generator import process_csv_to_png_pdf
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def example_single_file():
    """Example: Process a single CSV file."""
    csv_file = "results/stage_analysis/stage_analysis_2-5_daily_20250905.csv"

    if Path(csv_file).exists():
        logger.info(f"Processing: {csv_file}")
        png_path, pdf_path = process_csv_to_png_pdf(csv_file)
        logger.info(f"Generated: PNG={png_path}, PDF={pdf_path}")
    else:
        logger.warning(f"File not found: {csv_file}")

def example_batch_processing():
    """Example: Process multiple CSV files."""
    from report_generators.universal_png_pdf_generator import UniversalPngPdfGenerator

    generator = UniversalPngPdfGenerator()

    # Find CSV files
    csv_files = list(Path("results").rglob("*.csv"))[:5]  # Process first 5

    logger.info(f"Processing {len(csv_files)} CSV files...")

    for csv_file in csv_files:
        try:
            png_path, pdf_path = generator.process_csv_file(str(csv_file))
            logger.info(f"✅ {csv_file.name} → {Path(pdf_path).name}")
        except Exception as e:
            logger.error(f"❌ {csv_file.name}: {e}")

def example_custom_output_dir():
    """Example: Use custom output directory."""
    from report_generators.universal_png_pdf_generator import UniversalPngPdfGenerator

    # Custom output directory
    generator = UniversalPngPdfGenerator(output_dir_reports="custom_reports")

    csv_file = "results/stage_analysis/stage_analysis_2-5_daily_20250905.csv"

    if Path(csv_file).exists():
        png_path, pdf_path = generator.process_csv_file(csv_file)
        logger.info(f"Custom output: {pdf_path}")

if __name__ == "__main__":
    logger.info("🎯 Universal PNG-PDF Generator Examples")
    logger.info("=" * 50)

    # Example 1: Single file
    logger.info("\n1️⃣ Single File Processing:")
    example_single_file()

    # Example 2: Batch processing
    logger.info("\n2️⃣ Batch Processing:")
    example_batch_processing()

    # Example 3: Custom output directory
    logger.info("\n3️⃣ Custom Output Directory:")
    example_custom_output_dir()

    logger.info("\n✨ Examples completed!")