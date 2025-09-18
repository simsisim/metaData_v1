"""
Post-Process Workflow Orchestrator
=================================

Orchestrates the complete post-processing workflow:
1. Load configuration from user_data_pp.csv
2. Resolve logical filenames to physical file paths
3. Apply filters and sorts per configuration
4. Save processed CSV files
5. Generate reports (future enhancement)

This module does not modify existing files - it only creates new processed files.
"""

import pandas as pd
import os
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime

from .return_file_info import get_latest_file
from .generate_post_process_pdfs import generate_post_process_pdfs, extract_pdf_type_from_config, should_generate_pdf, is_pdf_enabled

logger = logging.getLogger(__name__)


class PostProcessWorkflow:
    """
    Handles the complete post-processing workflow for multiple files.
    """

    def __init__(self, config_path: str = "user_data_pp.csv", base_path: str = "."):
        """
        Initialize the workflow with configuration.

        Args:
            config_path: Path to the configuration CSV file
            base_path: Base directory path for file resolution
        """
        self.config_path = config_path
        self.base_path = base_path
        self.config_df = None
        self.processed_files = {}

    def load_configuration(self) -> bool:
        """
        Load the post-processing configuration from CSV.

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            config_file = Path(self.base_path) / self.config_path
            if not config_file.exists():
                logger.error(f"Configuration file not found: {config_file}")
                return False

            self.config_df = pd.read_csv(config_file)

            # Validate required columns (updated for new format)
            required_columns = ['Filename', 'Step', 'Action', 'Column']
            missing_columns = [col for col in required_columns if col not in self.config_df.columns]

            if missing_columns:
                logger.error(f"Missing required columns in config: {missing_columns}")
                return False

            # Skip comment rows (rows starting with #)
            self.config_df = self.config_df[~self.config_df['Filename'].astype(str).str.startswith('#')]

            logger.info(f"Loaded configuration with {len(self.config_df)} operations")
            return True

        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return False

    def apply_filters(self, df: pd.DataFrame, filter_ops: pd.DataFrame) -> pd.DataFrame:
        """
        Apply filter operations to a DataFrame using step-by-step boolean masking.

        Args:
            df: DataFrame to filter
            filter_ops: DataFrame containing filter operations

        Returns:
            Filtered DataFrame
        """
        if filter_ops.empty:
            return df

        result_df = df.copy()
        current_mask = pd.Series([True] * len(df), index=df.index)
        pending_or_masks = []

        for _, row in filter_ops.iterrows():
            col = row['Column']
            condition = row['Condition']
            value = row['Value']
            logic = row.get('Logic', 'AND')

            # Skip if any required field is missing
            if pd.isna(col) or pd.isna(condition) or pd.isna(value):
                continue

            if col not in df.columns:
                logger.warning(f"Filter column '{col}' not found in DataFrame")
                continue

            # Create condition mask
            try:
                condition_mask = pd.Series([False] * len(df), index=df.index)

                if condition.lower() == 'equals':
                    if str(value).lower() == 'true':
                        condition_mask = df[col] == True
                    elif str(value).lower() == 'false':
                        condition_mask = df[col] == False
                    else:
                        condition_mask = df[col] == value

                elif condition.lower() == 'greater_than':
                    condition_mask = df[col] > float(value)

                elif condition.lower() == 'less_than':
                    condition_mask = df[col] < float(value)

                elif condition.lower() == 'greater_equal':
                    condition_mask = df[col] >= float(value)

                elif condition.lower() == 'less_equal':
                    condition_mask = df[col] <= float(value)

                elif condition.lower() == 'contains':
                    condition_mask = df[col].astype(str).str.contains(str(value), na=False)

                elif condition.lower() == 'not_equals':
                    condition_mask = df[col] != value

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
        logger.info(f"Filter result: {len(df)} -> {len(filtered_df)} rows")
        return filtered_df

    def apply_sorts(self, df: pd.DataFrame, sort_ops: pd.DataFrame) -> pd.DataFrame:
        """
        Apply sort operations to a DataFrame.

        Args:
            df: DataFrame to sort
            sort_ops: DataFrame containing sort operations

        Returns:
            Sorted DataFrame
        """
        if sort_ops.empty:
            return df

        sort_columns = []
        ascending_flags = []

        for _, row in sort_ops.iterrows():
            col = row['Column']
            order = row.get('Order', 'asc')

            if pd.isna(col):
                continue

            if col not in df.columns:
                logger.warning(f"Sort column '{col}' not found in DataFrame")
                continue

            sort_columns.append(col)
            ascending_flags.append(order.lower() == 'asc')

        if sort_columns:
            try:
                sorted_df = df.sort_values(by=sort_columns, ascending=ascending_flags)
                logger.info(f"Applied sorting by: {sort_columns} (ascending: {ascending_flags})")
                return sorted_df
            except Exception as e:
                logger.error(f"Error applying sorts: {e}")
                return df
        else:
            return df

    def generate_output_filename(self, logical_filename: str, physical_path: str, file_ops: pd.DataFrame, file_id: str = None) -> str:
        """
        Generate output filename based on output_id or auto-generate from conditions.

        Args:
            logical_filename: Logical filename for operations lookup
            physical_path: Physical file path to extract original filename
            file_ops: DataFrame containing operations for this file
            file_id: File ID for unique output naming

        Returns:
            Generated filename string
        """
        # Get output_id from first row where it's defined
        output_id = None
        for _, row in file_ops.iterrows():
            if not pd.isna(row.get('Output_id')):
                output_id = str(row['Output_id']).strip()
                break

        if not output_id:
            # Auto-generate from filter conditions
            filter_ops = file_ops[file_ops['Action'].str.lower() == 'filter']
            conditions = []

            for _, row in filter_ops.iterrows():
                col = str(row['Column']) if not pd.isna(row.get('Column')) else ''
                condition = str(row['Condition']) if not pd.isna(row.get('Condition')) else ''
                value = str(row['Value']) if not pd.isna(row.get('Value')) else ''

                if col and condition and value:
                    # Create short condition descriptor
                    if condition.lower() == 'greater_than':
                        conditions.append(f"{col}_gt{value}")
                    elif condition.lower() == 'less_than':
                        conditions.append(f"{col}_lt{value}")
                    elif condition.lower() == 'equals':
                        if value.lower() == 'true':
                            conditions.append(f"{col}_true")
                        elif value.lower() == 'false':
                            conditions.append(f"{col}_false")
                        else:
                            conditions.append(f"{col}_{value}")
                    elif condition.lower() == 'contains':
                        conditions.append(f"{col}_has_{value}")
                    else:
                        conditions.append(f"{col}_{condition}_{value}")

            # Create auto-generated id
            if conditions:
                output_id = "_".join(conditions[:3])  # Limit to first 3 conditions
            else:
                output_id = "filtered"

        # Sanitize filename
        import re
        output_id = re.sub(r'[^a-zA-Z0-9_-]', '_', output_id)
        output_id = re.sub(r'_+', '_', output_id)  # Replace multiple underscores with single
        output_id = output_id.strip('_')  # Remove leading/trailing underscores

        # Extract original filename without extension from physical path
        from pathlib import Path
        original_filename = Path(physical_path).stem  # Gets filename without .csv extension

        # Include file_id in filename if provided
        if file_id is not None and str(file_id).strip() != '':
            return f"{original_filename}_pp_f{file_id}_{output_id}.csv"
        else:
            return f"{original_filename}_pp_{output_id}.csv"

    def process_file_group(self, logical_filename: str, file_id: str) -> Optional[str]:
        """
        Process a single file group (logical filename + file_id) through the complete workflow.

        Args:
            logical_filename: Logical filename to process
            file_id: File ID for this processing group

        Returns:
            Path to processed file, or None if failed
        """
        try:
            # 1. Resolve logical filename to physical path
            physical_path = get_latest_file(logical_filename, self.base_path)
            if not physical_path:
                logger.error(f"Could not resolve file: {logical_filename}")
                return None

            # 2. Load the data file
            logger.info(f"Loading: {physical_path}")
            df = pd.read_csv(physical_path)
            original_rows = len(df)

            # 3. Get operations for this file + file_id combination
            if 'File_id' in self.config_df.columns:
                file_ops = self.config_df[
                    (self.config_df['Filename'] == logical_filename) &
                    (self.config_df['File_id'].astype(str) == str(file_id))
                ].copy()
            else:
                # Fallback for configs without File_id column
                file_ops = self.config_df[self.config_df['Filename'] == logical_filename].copy()

            if file_ops.empty:
                logger.warning(f"No operations configured for: {logical_filename}, file_id: {file_id}")
                return None

            # Sort operations by step order
            file_ops = file_ops.sort_values('Step')

            # 4. Separate filter and sort operations
            filter_ops = file_ops[file_ops['Action'].str.lower() == 'filter']
            sort_ops = file_ops[file_ops['Action'].str.lower() == 'sort']

            # 5. Apply filters first
            if not filter_ops.empty:
                df = self.apply_filters(df, filter_ops)

            # 6. Apply sorts second
            if not sort_ops.empty:
                df = self.apply_sorts(df, sort_ops)

            # 7. Generate output filename
            output_filename = self.generate_output_filename(logical_filename, physical_path, file_ops, file_id)
            output_path = Path(self.base_path) / "results" / "post_process" / output_filename

            # Create output directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)

            df.to_csv(output_path, index=False)

            logger.info(f"Processed {logical_filename} (file_id: {file_id}): {original_rows} -> {len(df)} rows")
            logger.info(f"Saved to: {output_path}")

            # 8. Generate PDF if configured (sequential processing with in-memory DataFrame)
            pdf_path = None

            # Enhanced logging for PDF generation decision
            has_pdf_type = 'PDF_type' in file_ops.columns and not file_ops['PDF_type'].dropna().empty
            pdf_enabled = is_pdf_enabled(file_ops)

            if not pdf_enabled:
                logger.info(f"PDF generation disabled (PDF_enable=FALSE) for {logical_filename} (file_id: {file_id})")
            elif not has_pdf_type:
                logger.info(f"PDF generation skipped - no PDF_type specified for {logical_filename} (file_id: {file_id})")
            elif should_generate_pdf(file_ops):
                try:
                    pdf_type = extract_pdf_type_from_config(file_ops)
                    logger.info(f"Generating PDF (PDF_enable=TRUE) with template '{pdf_type}' for {logical_filename} (file_id: {file_id})")

                    # Create rich metadata context
                    metadata = {
                        'original_filename': logical_filename,
                        'file_id': file_id,
                        'physical_path': physical_path,
                        'original_rows': original_rows,
                        'filtered_rows': len(df),
                        'filter_operations': filter_ops.to_dict('records') if not filter_ops.empty else [],
                        'sort_operations': sort_ops.to_dict('records') if not sort_ops.empty else [],
                        'processing_timestamp': datetime.now(),
                        'dataframe_memory_usage': df.memory_usage(deep=True).sum(),
                        'columns_available': df.columns.tolist()
                    }

                    # Generate PDF with same filtered DataFrame
                    pdf_path = generate_post_process_pdfs(df, pdf_type, str(output_path), metadata)
                    logger.info(f"Successfully generated PDF: {pdf_path}")

                except Exception as pdf_error:
                    logger.warning(f"PDF generation failed for {logical_filename} (file_id: {file_id}): {pdf_error}")
                    # Continue processing - CSV is still successful

            return str(output_path)

        except Exception as e:
            logger.error(f"Error processing {logical_filename} (file_id: {file_id}): {e}")
            return None

    def run_workflow(self) -> Dict[str, str]:
        """
        Run the complete post-processing workflow for all configured file groups.

        Returns:
            Dict mapping (filename, file_id) combinations to processed file paths
        """
        results = {}

        try:
            # 1. Load configuration
            if not self.load_configuration():
                return results

            # 2. Get unique filename + file_id combinations to process
            if 'File_id' in self.config_df.columns:
                # Group by Filename + File_id
                file_groups = self.config_df.groupby(['Filename', 'File_id']).size().index.tolist()
                logger.info(f"Processing {len(file_groups)} file groups: {file_groups}")

                # 3. Process each file group
                for logical_filename, file_id in file_groups:
                    processed_path = self.process_file_group(logical_filename, str(file_id))
                    if processed_path:
                        group_key = f"{logical_filename}_f{file_id}"
                        results[group_key] = processed_path
                    else:
                        logger.error(f"Failed to process: {logical_filename}, file_id: {file_id}")
            else:
                # Fallback: process by filename only (backward compatibility)
                logical_filenames = self.config_df['Filename'].unique()
                logger.info(f"Processing {len(logical_filenames)} files (no File_id): {list(logical_filenames)}")

                for logical_filename in logical_filenames:
                    processed_path = self.process_file_group(logical_filename, '')
                    if processed_path:
                        results[logical_filename] = processed_path
                    else:
                        logger.error(f"Failed to process: {logical_filename}")

            logger.info(f"Workflow completed. Processed {len(results)} file groups successfully.")
            return results

        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            return results


def run_post_processing(config_path: str = "user_data_pp.csv", base_path: str = ".") -> Dict[str, str]:
    """
    Convenience function to run the complete post-processing workflow.

    Args:
        config_path: Path to configuration CSV
        base_path: Base directory path

    Returns:
        Dict mapping logical filenames to processed file paths
    """
    workflow = PostProcessWorkflow(config_path, base_path)
    return workflow.run_workflow()


if __name__ == "__main__":
    # Test the workflow
    import sys

    # Set up logging for testing
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    config_file = sys.argv[1] if len(sys.argv) > 1 else "user_data_pp.csv"

    print(f"Running post-processing workflow with config: {config_file}")
    results = run_post_processing(config_file)

    print("\nProcessing Results:")
    for logical_name, output_path in results.items():
        print(f"  {logical_name} -> {output_path}")

    if not results:
        print("  No files were processed successfully.")