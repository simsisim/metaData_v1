"""
Stage Analysis CSV Enricher

This module enriches stage analysis CSV files with business metadata from ticker_universe_all.csv
to enable grouping and filtering by sector, industry, index membership, and market cap.

Author: Claude Code
Date: 2025-09-18
"""

import pandas as pd
import os
import glob
from typing import List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StageAnalysisEnricher:
    """
    Enriches stage analysis CSV files with ticker universe metadata.
    """

    def __init__(self, ticker_universe_path: str, stage_analysis_dir: str):
        """
        Initialize the enricher with paths to ticker universe and stage analysis files.

        Args:
            ticker_universe_path: Path to ticker_universe_all.csv
            stage_analysis_dir: Directory containing stage analysis CSV files
        """
        self.ticker_universe_path = ticker_universe_path
        self.stage_analysis_dir = stage_analysis_dir
        self.ticker_universe_df = None

    def load_ticker_universe(self) -> bool:
        """
        Load the ticker universe data with metadata.

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not os.path.exists(self.ticker_universe_path):
                logger.error(f"Ticker universe file not found: {self.ticker_universe_path}")
                return False

            self.ticker_universe_df = pd.read_csv(self.ticker_universe_path)
            logger.info(f"Loaded ticker universe with {len(self.ticker_universe_df)} tickers")

            # Remove Description column as requested (check both cases)
            columns_to_remove = []
            for col in ['Description', 'description']:
                if col in self.ticker_universe_df.columns:
                    columns_to_remove.append(col)

            if columns_to_remove:
                self.ticker_universe_df = self.ticker_universe_df.drop(columns_to_remove, axis=1)
                logger.info(f"Removed columns: {columns_to_remove}")

            # Display available columns for enrichment
            enrichment_columns = [col for col in self.ticker_universe_df.columns if col.lower() not in ['ticker', 'description']]
            logger.info(f"Available enrichment columns: {enrichment_columns}")

            return True

        except Exception as e:
            logger.error(f"Error loading ticker universe: {str(e)}")
            return False

    def find_stage_analysis_files(self) -> List[str]:
        """
        Find all stage analysis CSV files in the directory.

        Returns:
            List[str]: List of stage analysis CSV file paths
        """
        if not os.path.exists(self.stage_analysis_dir):
            logger.error(f"Stage analysis directory not found: {self.stage_analysis_dir}")
            return []

        # Look for stage analysis CSV files
        patterns = [
            os.path.join(self.stage_analysis_dir, "stage_analysis_*.csv"),
            os.path.join(self.stage_analysis_dir, "**/stage_analysis_*.csv")
        ]

        files = []
        for pattern in patterns:
            files.extend(glob.glob(pattern, recursive=True))

        # Remove duplicates and sort
        files = sorted(list(set(files)))

        logger.info(f"Found {len(files)} stage analysis files")
        for file in files:
            logger.info(f"  - {os.path.basename(file)}")

        return files

    def enrich_stage_analysis_file(self, file_path: str) -> bool:
        """
        Enrich a single stage analysis CSV file with ticker universe metadata.

        Args:
            file_path: Path to the stage analysis CSV file

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load stage analysis data
            stage_df = pd.read_csv(file_path)
            original_columns = stage_df.columns.tolist()
            original_row_count = len(stage_df)

            logger.info(f"Processing {os.path.basename(file_path)} with {original_row_count} rows")

            # Check if ticker column exists (case insensitive)
            ticker_col = None
            for col in stage_df.columns:
                if col.lower() == 'ticker':
                    ticker_col = col
                    break

            if ticker_col is None:
                logger.error(f"No 'ticker' column found in {file_path}")
                return False

            # Rename columns to match for merge (use lowercase 'ticker')
            stage_df_merged = stage_df.copy()
            if ticker_col != 'ticker':
                stage_df_merged = stage_df_merged.rename(columns={ticker_col: 'ticker'})

            # Perform left join to add ticker universe metadata
            enriched_df = stage_df_merged.merge(
                self.ticker_universe_df,
                on='ticker',
                how='left'
            )

            # Restore original column name if it was different
            if ticker_col != 'ticker':
                enriched_df = enriched_df.rename(columns={'ticker': ticker_col})

            # Check for unmatched tickers
            unmatched_count = enriched_df.isnull().any(axis=1).sum()
            if unmatched_count > 0:
                unmatched_tickers = enriched_df[enriched_df.isnull().any(axis=1)][ticker_col].unique()
                logger.warning(f"Found {unmatched_count} rows with unmatched tickers: {list(unmatched_tickers[:5])}")

            # Get newly added columns
            new_columns = [col for col in enriched_df.columns if col not in original_columns]
            logger.info(f"Added {len(new_columns)} new columns: {new_columns}")

            # Overwrite the original file
            enriched_df.to_csv(file_path, index=False)
            logger.info(f"Successfully enriched {os.path.basename(file_path)}")

            return True

        except Exception as e:
            logger.error(f"Error enriching {file_path}: {str(e)}")
            return False

    def enrich_all_stage_analysis_files(self) -> bool:
        """
        Enrich all stage analysis CSV files with ticker universe metadata.

        Returns:
            bool: True if all files processed successfully, False otherwise
        """
        if self.ticker_universe_df is None:
            logger.error("Ticker universe data not loaded. Call load_ticker_universe() first.")
            return False

        files = self.find_stage_analysis_files()
        if not files:
            logger.warning("No stage analysis files found to enrich")
            return False

        success_count = 0
        total_files = len(files)

        for file_path in files:
            if self.enrich_stage_analysis_file(file_path):
                success_count += 1

        logger.info(f"Enrichment complete: {success_count}/{total_files} files processed successfully")
        return success_count == total_files


def enrich_stage_analysis_csvs(ticker_universe_path: str = None, stage_analysis_dir: str = None) -> bool:
    """
    Convenience function to enrich all stage analysis CSV files.

    Args:
        ticker_universe_path: Path to ticker_universe_all.csv (default: auto-detect)
        stage_analysis_dir: Directory with stage analysis files (default: auto-detect)

    Returns:
        bool: True if successful, False otherwise
    """
    # Auto-detect paths if not provided
    if ticker_universe_path is None:
        ticker_universe_path = "results/ticker_universes/ticker_universe_all.csv"

    if stage_analysis_dir is None:
        stage_analysis_dir = "results"

    # Create enricher instance
    enricher = StageAnalysisEnricher(ticker_universe_path, stage_analysis_dir)

    # Load ticker universe data
    if not enricher.load_ticker_universe():
        return False

    # Enrich all stage analysis files
    return enricher.enrich_all_stage_analysis_files()


if __name__ == "__main__":
    # Run enrichment when called directly
    success = enrich_stage_analysis_csvs()
    print(f"Stage analysis enrichment {'completed successfully' if success else 'failed'}")