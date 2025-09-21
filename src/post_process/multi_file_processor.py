"""
Multi-File Processor for Post-Processing Workflow
===============================================

Handles filtering and intersection operations across multiple data source files.
Supports complex filtering scenarios like:
- Performance metrics from basic_calculation files
- Stage analysis from stage_analysis files
- Fundamental data from financial_data files
- CANSLIM scores from canslim_screened files

Example use case:
Filter stocks with 22-day performance >20% AND bullish stage analysis
"""

import pandas as pd
import numpy as np
import logging
import gc
from typing import Dict, List, Set, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime

from .return_file_info import get_latest_file
from .generate_post_process_pdfs import generate_post_process_pdfs, extract_template_from_config, is_pdf_enabled

logger = logging.getLogger(__name__)


class MultiFileProcessor:
    """
    Handles multi-file filtering and intersection operations.
    """

    def __init__(self, config_df: pd.DataFrame, file_id: str, base_path: str = "."):
        """
        Initialize multi-file processor for a specific file_id.

        Args:
            config_df: Complete configuration DataFrame
            file_id: File ID to process
            base_path: Base directory path for file resolution
        """
        self.config_df = config_df
        self.file_id = file_id
        self.base_path = base_path
        self.source_files = {}
        self.filtered_results = {}
        self.ticker_sets = {}

        # Get config rows for this file_id
        self.file_config = self._get_file_config()

    def _get_file_config(self) -> pd.DataFrame:
        """Get configuration rows for this file_id."""
        if pd.isna(self.file_id) or self.file_id == '':
            return pd.DataFrame()

        return self.config_df[self.config_df['File_id'].astype(str) == str(self.file_id)].copy()

    def identify_source_files(self) -> Dict[str, str]:
        """
        Identify all source files needed for this file_id.

        Returns:
            Dict mapping source_file_type to logical_filename
        """
        source_mapping = {}

        for _, row in self.file_config.iterrows():
            filename = row.get('Filename', '')
            source_file = row.get('Source_File', '')

            if pd.notna(filename) and filename.strip():
                if pd.notna(source_file) and source_file.strip():
                    # Multi-file operation: use Source_File as key
                    source_mapping[source_file.strip()] = filename.strip()
                else:
                    # Default source: use filename as both key and value
                    source_mapping[filename.strip()] = filename.strip()

        logger.info(f"File_id {self.file_id} requires sources: {list(source_mapping.keys())}")
        return source_mapping

    def load_source_files(self) -> bool:
        """
        Load all required source files.

        Returns:
            True if all files loaded successfully, False otherwise
        """
        source_mapping = self.identify_source_files()

        for source_key, logical_filename in source_mapping.items():
            try:
                # Get latest physical file path
                file_path = get_latest_file(logical_filename, self.base_path)
                if not file_path:
                    logger.error(f"Could not find file for logical name: {logical_filename}")
                    return False

                # Load DataFrame
                df = pd.read_csv(file_path)
                self.source_files[source_key] = df
                logger.info(f"Loaded {source_key}: {file_path} ({len(df)} rows)")

            except Exception as e:
                logger.error(f"Error loading source file {logical_filename}: {e}")
                return False

        return True

    def apply_source_specific_filters(self) -> bool:
        """
        Apply filters to each source file independently.

        Returns:
            True if filtering successful, False otherwise
        """
        # Group operations by source file
        source_operations = {}

        for _, row in self.file_config.iterrows():
            filename = row.get('Filename', '')
            source_file = row.get('Source_File', '')
            action = row.get('Action', '').lower().strip()

            if action != 'filter':
                continue

            # Determine source key
            if pd.notna(source_file) and source_file.strip():
                source_key = source_file.strip()
            else:
                source_key = filename.strip()

            if source_key not in source_operations:
                source_operations[source_key] = []

            source_operations[source_key].append(row)

        # Apply filters per source
        for source_key, operations in source_operations.items():
            if source_key not in self.source_files:
                logger.error(f"Source file not loaded: {source_key}")
                return False

            df = self.source_files[source_key]
            filter_ops_df = pd.DataFrame(operations)

            # Apply filters using existing filter logic
            filtered_df = self._apply_filters_to_dataframe(df, filter_ops_df)
            self.filtered_results[source_key] = filtered_df

            # Extract ticker set
            if 'ticker' in filtered_df.columns:
                self.ticker_sets[source_key] = set(filtered_df['ticker'].unique())
            else:
                logger.warning(f"No 'ticker' column found in {source_key}")
                self.ticker_sets[source_key] = set()

            logger.info(f"Source {source_key}: {len(df)} -> {len(filtered_df)} rows, {len(self.ticker_sets[source_key])} unique tickers")

        return True

    def _apply_filters_to_dataframe(self, df: pd.DataFrame, filter_ops: pd.DataFrame) -> pd.DataFrame:
        """
        Apply filter operations to a DataFrame using the same logic as single-file processing.

        Args:
            df: DataFrame to filter
            filter_ops: DataFrame containing filter operations

        Returns:
            Filtered DataFrame
        """
        if filter_ops.empty:
            return df

        # Initialize mask to True for all rows
        current_mask = pd.Series([True] * len(df), index=df.index)
        pending_or_masks = []

        # Sort operations by Step to ensure proper order
        if 'Step' in filter_ops.columns:
            filter_ops = filter_ops.sort_values('Step')

        for _, row in filter_ops.iterrows():
            col = row.get('Column')
            condition = row.get('Condition')
            value = row.get('Value')
            logic = row.get('Logic', 'AND')

            # Skip if any required field is missing
            if pd.isna(col) or pd.isna(condition) or pd.isna(value):
                continue

            if col not in df.columns:
                logger.warning(f"Filter column '{col}' not found in DataFrame. Available columns: {list(df.columns)[:10]}...")
                continue

            # Create condition mask
            try:
                condition_mask = pd.Series([False] * len(df), index=df.index)

                # Convert condition to string and handle NaN/empty values
                condition_str = str(condition).strip().lower() if pd.notna(condition) else ''
                value_str = str(value).strip() if pd.notna(value) else ''

                if condition_str == 'equals':
                    if value_str.lower() == 'true':
                        condition_mask = df[col] == True
                    elif value_str.lower() == 'false':
                        condition_mask = df[col] == False
                    else:
                        condition_mask = df[col] == value

                elif condition_str == 'greater_than':
                    condition_mask = df[col] > float(value)

                elif condition_str == 'less_than':
                    condition_mask = df[col] < float(value)

                elif condition_str == 'greater_equal':
                    condition_mask = df[col] >= float(value)

                elif condition_str == 'less_equal':
                    condition_mask = df[col] <= float(value)

                elif condition_str == 'contains':
                    condition_mask = df[col].astype(str).str.contains(value_str, na=False, case=False)

                elif condition_str == 'not_equals':
                    condition_mask = df[col] != value

                elif condition_str == 'between':
                    # Handle between operation: value should be "min,max"
                    try:
                        min_val, max_val = map(float, value_str.split(','))
                        condition_mask = (df[col] >= min_val) & (df[col] <= max_val)
                    except:
                        logger.warning(f"Invalid between format: {value_str}. Use 'min,max'")
                        continue

                else:
                    logger.warning(f"Unknown condition: {condition}")
                    continue

                # Apply logic
                if pd.isna(logic) or logic.upper() == 'AND':
                    # Process any pending OR operations first
                    if pending_or_masks:
                        or_combined = pd.Series([False] * len(df), index=df.index)
                        for or_mask in pending_or_masks:
                            or_combined = or_combined | or_mask
                        current_mask = current_mask & or_combined
                        pending_or_masks = []

                    # Apply AND logic
                    current_mask = current_mask & condition_mask

                elif logic.upper() == 'OR':
                    # Collect OR conditions
                    pending_or_masks.append(condition_mask)

                logger.debug(f"Applied filter: {col} {condition} {value} ({condition_mask.sum()} matches)")

            except Exception as e:
                logger.warning(f"Error applying filter {col} {condition} {value}: {e}")
                continue

        # Process any remaining OR operations
        if pending_or_masks:
            or_combined = pd.Series([False] * len(df), index=df.index)
            for or_mask in pending_or_masks:
                or_combined = or_combined | or_mask
            current_mask = current_mask & or_combined

        # Apply final mask
        filtered_df = df[current_mask]
        return filtered_df

    def perform_intersection_operations(self) -> Set[str]:
        """
        Perform intersection/union operations on ticker sets.

        Returns:
            Final set of tickers after all operations
        """
        if not self.ticker_sets:
            logger.warning("No ticker sets available for intersection")
            return set()

        source_keys = list(self.ticker_sets.keys())

        if len(source_keys) == 1:
            # Single source, return as-is
            final_tickers = self.ticker_sets[source_keys[0]]
        else:
            # Multiple sources, perform intersection (AND logic by default)
            final_tickers = set.intersection(*self.ticker_sets.values())

        logger.info(f"Intersection result: {len(final_tickers)} tickers from {len(source_keys)} sources")
        logger.info(f"Source ticker counts: {[(k, len(v)) for k, v in self.ticker_sets.items()]}")

        return final_tickers

    def merge_multi_source_data(self, final_tickers: Set[str]) -> pd.DataFrame:
        """
        Merge data from all sources for the final ticker set.

        Args:
            final_tickers: Set of tickers to include in final result

        Returns:
            Combined DataFrame with data from all sources
        """
        if not final_tickers:
            logger.warning("No tickers in final set, returning empty DataFrame")
            return pd.DataFrame()

        # Start with the first source as base
        source_keys = list(self.filtered_results.keys())
        if not source_keys:
            return pd.DataFrame()

        # Use first source as base
        base_key = source_keys[0]
        base_df = self.filtered_results[base_key]
        result_df = base_df[base_df['ticker'].isin(final_tickers)].copy()

        # Add source identifier
        result_df[f'source_{base_key}'] = True

        # Merge additional sources
        for source_key in source_keys[1:]:
            source_df = self.filtered_results[source_key]
            source_subset = source_df[source_df['ticker'].isin(final_tickers)].copy()

            if source_subset.empty:
                continue

            # Add source identifier
            source_subset[f'source_{source_key}'] = True

            # Merge on ticker, handling column conflicts
            merge_cols = ['ticker']

            # Get unique columns from source (avoid duplicates)
            unique_cols = []
            for col in source_subset.columns:
                if col not in result_df.columns or col == 'ticker':
                    unique_cols.append(col)
                else:
                    # Rename conflicting columns
                    new_col_name = f"{col}_{source_key}"
                    source_subset = source_subset.rename(columns={col: new_col_name})
                    unique_cols.append(new_col_name)

            source_subset = source_subset[unique_cols]

            # Perform merge
            result_df = pd.merge(result_df, source_subset, on='ticker', how='left')

        # Fill missing source identifiers
        for source_key in source_keys:
            source_col = f'source_{source_key}'
            if source_col in result_df.columns:
                result_df[source_col] = result_df[source_col].fillna(False)

        logger.info(f"Merged data for {len(result_df)} tickers from {len(source_keys)} sources")
        return result_df

    def generate_output_filename(self) -> str:
        """
        Generate output filename for multi-file result.

        Returns:
            Generated filename string
        """
        # Get output_id from configuration
        output_id = None
        for _, row in self.file_config.iterrows():
            if not pd.isna(row.get('Output_id')):
                output_id = str(row['Output_id']).strip()
                break

        if not output_id:
            # Auto-generate from source files
            source_keys = list(self.ticker_sets.keys())
            output_id = f"multi_file_{'_'.join(source_keys[:3])}"  # Limit to first 3 sources

        # Add timestamp and multi-file identifier
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"multi_file_{output_id}_{timestamp}.csv"

    def process(self) -> Optional[str]:
        """
        Execute complete multi-file processing workflow.

        Returns:
            Path to generated output file, or None if failed
        """
        try:
            logger.info(f"Starting multi-file processing for File_id: {self.file_id}")

            # 1. Load all source files
            if not self.load_source_files():
                logger.error("Failed to load source files")
                return None

            # 2. Apply filters to each source
            if not self.apply_source_specific_filters():
                logger.error("Failed to apply source-specific filters")
                return None

            # 3. Perform intersection operations
            final_tickers = self.perform_intersection_operations()
            if not final_tickers:
                logger.warning("No tickers remain after intersection")
                # Still create empty file for consistency

            # 4. Merge data from all sources
            result_df = self.merge_multi_source_data(final_tickers)

            # 5. Generate output filename and save
            output_filename = self.generate_output_filename()
            output_dir = Path(self.base_path) / "results" / "post_process"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / output_filename

            result_df.to_csv(output_path, index=False)
            logger.info(f"Multi-file result saved: {output_path} ({len(result_df)} rows)")

            # 6. Generate PDF if configured
            if self._should_generate_pdf():
                try:
                    pdf_path = self._generate_pdf(result_df, str(output_path))
                    if pdf_path:
                        logger.info(f"Multi-file PDF generated: {pdf_path}")
                    else:
                        logger.warning(f"PDF generation returned no path for File_id: {self.file_id}")
                except Exception as pdf_error:
                    logger.warning(f"PDF generation failed for multi-file File_id {self.file_id}: {pdf_error}")
                    # Continue - CSV processing was successful

            # 7. Memory cleanup: explicitly delete all DataFrames and force garbage collection
            self._cleanup_memory(result_df)

            return str(output_path)

        except Exception as e:
            logger.error(f"Error in multi-file processing: {e}")
            return None

    def _should_generate_pdf(self) -> bool:
        """
        Determine if PDF generation should be performed for this multi-file configuration.

        Returns:
            True if PDF should be generated, False otherwise
        """
        try:
            # Use primary source selection logic for PDF configuration
            primary_config = self._get_primary_source_config()

            if primary_config.empty:
                logger.debug(f"No primary source config found for PDF generation (File_id: {self.file_id})")
                return False

            # Check if PDF generation is enabled
            pdf_enabled = is_pdf_enabled(primary_config)

            # Check if template/PDF_type is specified
            has_template_config = (
                ('Template' in primary_config.columns and not primary_config['Template'].dropna().empty) or
                ('PDF_type' in primary_config.columns and not primary_config['PDF_type'].dropna().empty)
            )

            result = pdf_enabled and has_template_config
            logger.debug(f"PDF generation check for File_id {self.file_id}: enabled={pdf_enabled}, has_template={has_template_config}, result={result}")

            return result

        except Exception as e:
            logger.warning(f"Error checking PDF configuration for File_id {self.file_id}: {e}")
            return False

    def _get_primary_source_config(self) -> pd.DataFrame:
        """
        Get primary source configuration for PDF settings using precedence rules.

        Priority order:
        1. basic_calculation rows (primary business data)
        2. stage_analysis rows (secondary analysis)
        3. Other source types (if any)

        Returns:
            DataFrame with primary source configuration
        """
        if self.file_config.empty:
            return pd.DataFrame()

        # Priority 1: basic_calculation rows
        basic_calc_rows = self.file_config[
            self.file_config['Filename'].str.contains('basic_calculation', case=False, na=False)
        ]
        if not basic_calc_rows.empty:
            logger.debug(f"Using basic_calculation as primary source for PDF config (File_id: {self.file_id})")
            return basic_calc_rows

        # Priority 2: stage_analysis rows
        stage_rows = self.file_config[
            self.file_config['Filename'].str.contains('stage_analysis', case=False, na=False)
        ]
        if not stage_rows.empty:
            logger.debug(f"Using stage_analysis as primary source for PDF config (File_id: {self.file_id})")
            return stage_rows

        # Priority 3: Any other rows
        logger.debug(f"Using first available source as primary for PDF config (File_id: {self.file_id})")
        return self.file_config.head(1)

    def _generate_pdf(self, result_df: pd.DataFrame, csv_path: str) -> Optional[str]:
        """
        Generate PDF from multi-file result using appropriate template.

        Args:
            result_df: Combined DataFrame from multi-file processing
            csv_path: Path to the generated CSV file

        Returns:
            Path to generated PDF file, or None if failed
        """
        try:
            # Get primary source configuration for PDF settings
            primary_config = self._get_primary_source_config()

            if primary_config.empty:
                logger.error(f"No primary source configuration found for PDF generation (File_id: {self.file_id})")
                return None

            # Extract template name using existing logic
            template_name = extract_template_from_config(primary_config)
            logger.info(f"Multi-file PDF generation using template '{template_name}' for File_id: {self.file_id}")

            # Build comprehensive metadata for template consumption
            metadata = self._build_pdf_metadata(primary_config)

            # Generate PDF using existing infrastructure
            pdf_path = generate_post_process_pdfs(result_df, template_name, csv_path, metadata)

            return pdf_path

        except Exception as e:
            logger.error(f"Error generating PDF for multi-file File_id {self.file_id}: {e}")
            return None

    def _build_pdf_metadata(self, primary_config: pd.DataFrame) -> dict:
        """
        Build comprehensive metadata context for PDF template consumption.

        Args:
            primary_config: Primary source configuration DataFrame

        Returns:
            Metadata dictionary for PDF templates
        """
        try:
            # Aggregate filter operations from all sources
            filter_operations = []
            for _, row in self.file_config.iterrows():
                if row.get('Action', '').lower().strip() == 'filter':
                    filter_op = {
                        'Column': row.get('Column'),
                        'Condition': row.get('Condition'),
                        'Value': row.get('Value'),
                        'Logic': row.get('Logic', 'AND'),
                        'Source': row.get('Filename', 'unknown')
                    }
                    filter_operations.append(filter_op)

            # Calculate intersection statistics
            intersection_stats = {
                'source_count': len(self.ticker_sets),
                'source_names': list(self.ticker_sets.keys()),
                'individual_counts': {k: len(v) for k, v in self.ticker_sets.items()},
                'final_count': len(self.ticker_sets[list(self.ticker_sets.keys())[0]]) if self.ticker_sets else 0
            }

            if len(self.ticker_sets) > 1:
                intersection_stats['intersection_rate'] = (
                    intersection_stats['final_count'] /
                    min(intersection_stats['individual_counts'].values())
                ) if intersection_stats['individual_counts'] else 0

            metadata = {
                'file_id': self.file_id,
                'processing_mode': 'multi_file',
                'source_files': list(self.source_files.keys()),
                'primary_source': primary_config['Filename'].iloc[0] if not primary_config.empty else 'unknown',
                'filter_operations': filter_operations,
                'intersection_stats': intersection_stats,
                'generation_timestamp': datetime.now().isoformat(),
                'total_rows': len(self.source_files[list(self.source_files.keys())[0]]) if self.source_files else 0
            }

            logger.debug(f"Built PDF metadata for File_id {self.file_id}: {len(filter_operations)} filters, {intersection_stats['source_count']} sources")
            return metadata

        except Exception as e:
            logger.warning(f"Error building PDF metadata for File_id {self.file_id}: {e}")
            return {
                'file_id': self.file_id,
                'processing_mode': 'multi_file',
                'error': str(e)
            }

    def _cleanup_memory(self, result_df: pd.DataFrame) -> None:
        """
        Comprehensive memory cleanup for all DataFrames used in multi-file processing.

        Args:
            result_df: The final merged DataFrame to be cleaned up
        """
        try:
            total_memory_released = 0.0

            # Calculate memory usage before cleanup
            if hasattr(result_df, 'memory_usage'):
                result_memory = result_df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
                total_memory_released += result_memory

            # Clean up source files DataFrames
            source_memory = 0.0
            for source_key, df in self.source_files.items():
                if hasattr(df, 'memory_usage'):
                    source_memory += df.memory_usage(deep=True).sum() / 1024 / 1024  # MB

            # Clean up filtered results DataFrames
            filtered_memory = 0.0
            for source_key, df in self.filtered_results.items():
                if hasattr(df, 'memory_usage'):
                    filtered_memory += df.memory_usage(deep=True).sum() / 1024 / 1024  # MB

            total_memory_released += source_memory + filtered_memory

            # Explicit cleanup
            del result_df
            self.source_files.clear()
            self.filtered_results.clear()
            self.ticker_sets.clear()

            # Force garbage collection
            gc.collect()

            logger.info(f"Memory cleanup completed for File_id {self.file_id}: "
                       f"Released ~{total_memory_released:.2f} MB "
                       f"(result: {result_memory:.2f} MB, sources: {source_memory:.2f} MB, "
                       f"filtered: {filtered_memory:.2f} MB)")

        except Exception as e:
            logger.warning(f"Error during memory cleanup for File_id {self.file_id}: {e}")
            # Still attempt basic cleanup
            try:
                del result_df
                self.source_files.clear()
                self.filtered_results.clear()
                self.ticker_sets.clear()
                gc.collect()
            except:
                pass


def determine_processing_mode(config_df: pd.DataFrame, file_id: str) -> str:
    """
    Determine if file_id requires single-file or multi-file processing.

    Args:
        config_df: Configuration DataFrame
        file_id: File ID to analyze

    Returns:
        "single_file" or "multi_file"
    """
    if pd.isna(file_id) or file_id == '':
        return "single_file"

    file_config = config_df[config_df['File_id'].astype(str) == str(file_id)]
    if file_config.empty:
        return "single_file"

    # Check for Source_File column usage (if it exists)
    source_files_count = 0
    if 'Source_File' in file_config.columns:
        source_files = file_config['Source_File'].dropna()
        source_files_count = len(source_files)

    unique_filenames = file_config['Filename'].dropna().nunique()

    # Multi-file if:
    # 1. Source_File column is used, OR
    # 2. Multiple different filenames for same file_id
    if source_files_count > 0 or unique_filenames > 1:
        return "multi_file"
    else:
        return "single_file"