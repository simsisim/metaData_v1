"""
Data Transformer
================

Core transformation logic that combines all processing steps.
Handles individual file processing using origin-specific strategies.
Routes processing based on file_origin to appropriate transformation strategy.
"""

import pandas as pd
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

from .factories.transformation_factory import TransformationFactory

logger = logging.getLogger(__name__)


class DataTransformer:
    """
    Core data transformation engine that processes individual files.
    Uses origin-specific strategies for different data sources (TW, GF, etc.).
    """

    def __init__(self, config):
        self.config = config
        self.processing_stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'errors': [],
            'warnings': [],
            'origins_processed': {},
            'strategy_stats': {}
        }

    def transform_file(self, file_info: Dict) -> Dict:
        """
        Transform a single file using origin-specific strategy.

        Args:
            file_info: File information dictionary from FilePatternMatcher
                      Must contain 'rule' with 'file_origin' field

        Returns:
            Dict: Processing result with success status and details
        """
        result = {
            'success': False,
            'source_file': file_info['source_path'],
            'target_file': None,
            'ticker': None,
            'file_origin': None,
            'processing_info': {},
            'error': None
        }

        try:
            # Get file origin from rule
            rule = file_info.get('rule', {})
            file_origin = rule.get('file_origin')

            if not file_origin:
                result['error'] = "Missing file_origin in rule configuration"
                return result

            result['file_origin'] = file_origin

            # Get appropriate transformation strategy
            try:
                strategy = TransformationFactory.get_strategy(file_origin)
                logger.debug(f"Using {file_origin} strategy for file: {file_info['filename']}")
            except ValueError as e:
                result['error'] = f"Strategy creation failed: {e}"
                return result

            # Extract ticker using strategy
            ticker = strategy.extract_ticker(file_info['filename'])
            if not ticker:
                result['error'] = f"Could not extract ticker from filename: {file_info['filename']}"
                return result

            result['ticker'] = ticker

            # Load source file
            try:
                df = pd.read_csv(file_info['source_path'])
                logger.debug(f"Loaded source file: {file_info['source_path']} ({len(df)} rows)")
            except Exception as e:
                result['error'] = f"Failed to load source file: {e}"
                return result

            # Standardize columns using strategy
            standardized_df, processing_info = strategy.standardize_columns(df)
            result['processing_info'] = processing_info

            # Check for critical errors
            if processing_info.get('errors'):
                result['error'] = f"Standardization errors: {'; '.join(processing_info['errors'])}"
                return result

            # Generate target filename
            target_filename = self.config.resolve_target_filename(ticker, file_info['pattern'])
            target_path = os.path.join(file_info['target_folder'], target_filename)
            result['target_file'] = target_path

            # Ensure target directory exists
            Path(file_info['target_folder']).mkdir(parents=True, exist_ok=True)

            # Save standardized file
            try:
                standardized_df.to_csv(target_path, index=False)
                logger.info(f"[{file_origin}] Saved standardized file: {target_path}")
            except Exception as e:
                result['error'] = f"Failed to save target file: {e}"
                return result

            result['success'] = True

            # Update strategy stats
            strategy.update_stats(success=True)

            # Update overall stats
            self.processing_stats['successful'] += 1
            self._update_origin_stats(file_origin, success=True)

            # Log warnings if any
            if processing_info.get('warnings'):
                warning_msg = f"[{file_origin}] Warnings for {file_info['filename']}: {'; '.join(processing_info['warnings'])}"
                logger.warning(warning_msg)
                self.processing_stats['warnings'].append(warning_msg)
                strategy.update_stats(success=True, warning=warning_msg)

        except Exception as e:
            result['error'] = f"Unexpected error processing file: {e}"
            logger.error(result['error'])

            # Update stats for failure
            if result.get('file_origin'):
                self._update_origin_stats(result['file_origin'], success=False)

        finally:
            self.processing_stats['total_processed'] += 1
            if not result['success']:
                self.processing_stats['failed'] += 1
                if result['error']:
                    self.processing_stats['errors'].append(result['error'])

        return result

    def transform_batch(self, file_batch: List[Dict]) -> List[Dict]:
        """
        Transform a batch of files.

        Args:
            file_batch: List of file information dictionaries

        Returns:
            List[Dict]: List of processing results
        """
        results = []
        batch_size = len(file_batch)

        logger.info(f"Processing batch of {batch_size} files")

        for i, file_info in enumerate(file_batch, 1):
            logger.debug(f"Processing file {i}/{batch_size}: {file_info['filename']}")

            result = self.transform_file(file_info)
            results.append(result)

            # Log progress
            if result['success']:
                logger.debug(f"✓ Successfully processed: {file_info['filename']}")
            else:
                logger.warning(f"✗ Failed to process: {file_info['filename']} - {result['error']}")

        return results

    def _update_origin_stats(self, file_origin: str, success: bool):
        """Update processing statistics by origin."""
        if file_origin not in self.processing_stats['origins_processed']:
            self.processing_stats['origins_processed'][file_origin] = {
                'total': 0,
                'successful': 0,
                'failed': 0
            }

        origin_stats = self.processing_stats['origins_processed'][file_origin]
        origin_stats['total'] += 1

        if success:
            origin_stats['successful'] += 1
        else:
            origin_stats['failed'] += 1

    def validate_source_file(self, file_path: str, file_origin: str = None) -> Dict:
        """
        Validate a source file before processing using appropriate strategy.

        Args:
            file_path: Path to source file
            file_origin: Origin identifier for strategy selection

        Returns:
            Dict: Validation result
        """
        validation = {
            'is_valid': False,
            'errors': [],
            'warnings': [],
            'file_info': {}
        }

        try:
            # Check file exists
            if not os.path.exists(file_path):
                validation['errors'].append("File does not exist")
                return validation

            # Check file accessibility
            if not os.access(file_path, os.R_OK):
                validation['errors'].append("File is not readable")
                return validation

            # Get file info
            file_size = os.path.getsize(file_path)
            validation['file_info']['size_bytes'] = file_size

            if file_size == 0:
                validation['errors'].append("File is empty")
                return validation

            # Try to load and validate DataFrame
            try:
                df = pd.read_csv(file_path)
                validation['file_info']['rows'] = len(df)
                validation['file_info']['columns'] = list(df.columns)

                # Use strategy-specific validation if origin provided
                if file_origin:
                    try:
                        strategy = TransformationFactory.get_strategy(file_origin)
                        format_validation = strategy.validate_format(df)

                        validation['is_valid'] = format_validation['is_valid']
                        validation['errors'].extend(format_validation.get('errors', []))
                        validation['warnings'].extend(format_validation.get('warnings', []))

                    except ValueError:
                        validation['warnings'].append(f"Unknown file origin: {file_origin}")
                else:
                    # Generic validation - check if it's at least a readable CSV
                    validation['is_valid'] = True
                    validation['warnings'].append("No file origin specified - generic validation only")

            except Exception as e:
                validation['errors'].append(f"Failed to load CSV: {e}")

        except Exception as e:
            validation['errors'].append(f"Validation error: {e}")

        return validation

    def get_processing_stats(self) -> Dict:
        """Get current processing statistics."""
        return self.processing_stats.copy()

    def reset_stats(self):
        """Reset processing statistics."""
        self.processing_stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'errors': [],
            'warnings': [],
            'origins_processed': {},
            'strategy_stats': {}
        }

    def generate_processing_report(self) -> str:
        """Generate a summary report of processing results."""
        stats = self.processing_stats

        report_lines = [
            "=== PRE_PROCESS Data Transformation Report ===",
            f"Total files processed: {stats['total_processed']}",
            f"Successful: {stats['successful']}",
            f"Failed: {stats['failed']}",
            f"Success rate: {(stats['successful'] / max(stats['total_processed'], 1)) * 100:.1f}%"
        ]

        # Add origin-specific statistics
        if stats['origins_processed']:
            report_lines.extend([
                "",
                "=== Processing by Origin ==="
            ])

            for origin, origin_stats in stats['origins_processed'].items():
                success_rate = (origin_stats['successful'] / max(origin_stats['total'], 1)) * 100
                report_lines.append(
                    f"{origin}: {origin_stats['successful']}/{origin_stats['total']} "
                    f"({success_rate:.1f}% success)"
                )

        if stats['warnings']:
            report_lines.extend([
                "",
                f"Warnings ({len(stats['warnings'])}):",
                *[f"  - {warning}" for warning in stats['warnings'][:10]]  # Limit to first 10
            ])

        if stats['errors']:
            report_lines.extend([
                "",
                f"Errors ({len(stats['errors'])}):",
                *[f"  - {error}" for error in stats['errors'][:10]]  # Limit to first 10
            ])

        return "\n".join(report_lines)

    def get_transformation_summary(self, results: List[Dict]) -> Dict:
        """
        Get summary of transformation results.

        Args:
            results: List of processing results

        Returns:
            Dict: Summary statistics
        """
        summary = {
            'total_files': len(results),
            'successful': sum(1 for r in results if r['success']),
            'failed': sum(1 for r in results if not r['success']),
            'processed_tickers': list(set(r['ticker'] for r in results if r['ticker'])),
            'target_files': [r['target_file'] for r in results if r['success']],
            'error_files': [r['source_file'] for r in results if not r['success']],
            'origins_summary': {}
        }

        # Add origin-specific summary
        for result in results:
            origin = result.get('file_origin', 'Unknown')
            if origin not in summary['origins_summary']:
                summary['origins_summary'][origin] = {
                    'total': 0,
                    'successful': 0,
                    'failed': 0,
                    'tickers': set()
                }

            summary['origins_summary'][origin]['total'] += 1
            if result['success']:
                summary['origins_summary'][origin]['successful'] += 1
                if result.get('ticker'):
                    summary['origins_summary'][origin]['tickers'].add(result['ticker'])
            else:
                summary['origins_summary'][origin]['failed'] += 1

        # Convert ticker sets to lists
        for origin_data in summary['origins_summary'].values():
            origin_data['tickers'] = list(origin_data['tickers'])

        summary['success_rate'] = (summary['successful'] / max(summary['total_files'], 1)) * 100

        return summary