"""
PRE_PROCESS Manager
==================

Main orchestrator for the PRE_PROCESS module.
Coordinates configuration loading, file discovery, and data transformation.
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional
import logging

from .pre_process_config import PreProcessConfig
from .file_pattern_matcher import FilePatternMatcher
from .data_transformer import DataTransformer

logger = logging.getLogger(__name__)


class PreProcessManager:
    """
    Main orchestrator for the PRE_PROCESS module.
    Handles the complete workflow from configuration to file transformation.
    """

    def __init__(self, config_file: str = "user_data_pre_process.csv"):
        self.config_file = config_file
        self.config = None
        self.file_matcher = None
        self.data_transformer = None
        self.processing_results = {}

    def run_pre_processing(self) -> Dict:
        """
        Execute the complete PRE_PROCESS workflow.

        Returns:
            Dict: Processing results and statistics
        """
        workflow_start_time = time.time()

        logger.info("="*60)
        logger.info("Starting PRE_PROCESS workflow")
        logger.info("="*60)

        try:
            # Step 1: Load configuration
            config_result = self._load_configuration()
            if not config_result['success']:
                return self._create_error_result("Configuration loading failed", config_result)

            # Step 2: Discover files
            discovery_result = self._discover_files()
            if not discovery_result['success']:
                return self._create_error_result("File discovery failed", discovery_result)

            # Step 3: Transform files
            transformation_result = self._transform_files()
            if not transformation_result['success']:
                return self._create_error_result("File transformation failed", transformation_result)

            # Step 4: Generate final report
            workflow_duration = time.time() - workflow_start_time
            final_result = self._generate_final_report(workflow_duration)

            logger.info("PRE_PROCESS workflow completed successfully")
            logger.info("="*60)

            return final_result

        except Exception as e:
            error_msg = f"Unexpected error in PRE_PROCESS workflow: {e}"
            logger.error(error_msg)
            return self._create_error_result(error_msg, {'exception': str(e)})

    def _load_configuration(self) -> Dict:
        """Load and validate PRE_PROCESS configuration."""
        logger.info("Step 1: Loading PRE_PROCESS configuration")

        try:
            # Load configuration
            self.config = PreProcessConfig(self.config_file)

            # Check if PRE_PROCESS is enabled
            if not self.config.is_enabled():
                return {
                    'success': False,
                    'error': 'PRE_PROCESS is disabled (no valid processing rules found)'
                }

            # Log configuration summary
            config_summary = self.config.get_config_summary()
            logger.info(f"Configuration loaded: {config_summary['total_rules']} processing rules")

            for rule in self.config.get_processing_rules():
                logger.info(f"  Rule: {rule['pattern_match']} -> {rule['target_folder']}")

            return {'success': True, 'config_summary': config_summary}

        except Exception as e:
            error_msg = f"Failed to load configuration: {e}"
            logger.error(error_msg)
            return {'success': False, 'error': error_msg}

    def _discover_files(self) -> Dict:
        """Discover files matching configured patterns."""
        logger.info("Step 2: Discovering files for processing")

        try:
            # Initialize file matcher
            self.file_matcher = FilePatternMatcher(self.config)

            # Discover files
            discovered_files = self.file_matcher.discover_files()

            if not discovered_files:
                return {
                    'success': False,
                    'error': 'No files found matching configured patterns'
                }

            # Log discovery summary
            discovery_summary = self.file_matcher.get_processing_summary()
            logger.info(f"File discovery completed: {discovery_summary['total_files']} files found")

            for pattern, count in discovery_summary['file_counts_by_pattern'].items():
                logger.info(f"  Pattern '{pattern}': {count} files")

            return {
                'success': True,
                'discovered_files': discovered_files,
                'discovery_summary': discovery_summary
            }

        except Exception as e:
            error_msg = f"Failed to discover files: {e}"
            logger.error(error_msg)
            return {'success': False, 'error': error_msg}

    def _transform_files(self) -> Dict:
        """Transform discovered files from TradingView to standard format."""
        logger.info("Step 3: Transforming files")

        try:
            # Initialize data transformer
            self.data_transformer = DataTransformer(self.config)

            # Get discovered files
            discovered_files = self.file_matcher.get_processing_summary()['total_files']
            if discovered_files == 0:
                return {
                    'success': False,
                    'error': 'No files to transform'
                }

            # Prepare batches for processing
            batches = self.file_matcher.prepare_batch_processing(batch_size=10)
            logger.info(f"Processing {len(batches)} batches of files")

            all_results = []
            batch_count = 0

            # Process each batch
            for batch in batches:
                batch_count += 1
                logger.info(f"Processing batch {batch_count}/{len(batches)} ({len(batch)} files)")

                batch_results = self.data_transformer.transform_batch(batch)
                all_results.extend(batch_results)

                # Log batch summary
                successful_in_batch = sum(1 for r in batch_results if r['success'])
                logger.info(f"Batch {batch_count} completed: {successful_in_batch}/{len(batch)} successful")

            # Generate transformation summary
            transformation_summary = self.data_transformer.get_transformation_summary(all_results)
            logger.info(f"File transformation completed: {transformation_summary['successful']}/{transformation_summary['total_files']} successful")

            return {
                'success': True,
                'transformation_results': all_results,
                'transformation_summary': transformation_summary,
                'processing_stats': self.data_transformer.get_processing_stats()
            }

        except Exception as e:
            error_msg = f"Failed to transform files: {e}"
            logger.error(error_msg)
            return {'success': False, 'error': error_msg}

    def _generate_final_report(self, workflow_duration: float) -> Dict:
        """Generate final processing report."""
        logger.info("Step 4: Generating final report")

        try:
            # Collect all results
            config_summary = self.config.get_config_summary()
            discovery_summary = self.file_matcher.get_processing_summary()
            processing_stats = self.data_transformer.get_processing_stats()

            # Generate detailed report
            report = self.data_transformer.generate_processing_report()
            logger.info(f"Processing report generated:\n{report}")

            final_result = {
                'success': True,
                'workflow_duration': workflow_duration,
                'config_summary': config_summary,
                'discovery_summary': discovery_summary,
                'processing_stats': processing_stats,
                'detailed_report': report
            }

            # Log final summary
            logger.info(f"PRE_PROCESS workflow summary:")
            logger.info(f"  Duration: {workflow_duration:.2f} seconds")
            logger.info(f"  Files processed: {processing_stats['total_processed']}")
            logger.info(f"  Success rate: {(processing_stats['successful'] / max(processing_stats['total_processed'], 1)) * 100:.1f}%")

            return final_result

        except Exception as e:
            error_msg = f"Failed to generate final report: {e}"
            logger.error(error_msg)
            return {'success': False, 'error': error_msg}

    def _create_error_result(self, error_message: str, details: Dict) -> Dict:
        """Create standardized error result."""
        return {
            'success': False,
            'error': error_message,
            'details': details,
            'workflow_duration': 0
        }

    def get_processing_status(self) -> Dict:
        """Get current processing status."""
        status = {
            'config_loaded': self.config is not None,
            'files_discovered': self.file_matcher is not None,
            'transformation_active': self.data_transformer is not None
        }

        if self.config:
            status['config_summary'] = self.config.get_config_summary()

        if self.file_matcher:
            status['discovery_summary'] = self.file_matcher.get_processing_summary()

        if self.data_transformer:
            status['processing_stats'] = self.data_transformer.get_processing_stats()

        return status

    def validate_configuration(self) -> Dict:
        """Validate PRE_PROCESS configuration without running workflow."""
        try:
            config = PreProcessConfig(self.config_file)
            config_summary = config.get_config_summary()

            validation_result = {
                'is_valid': config.is_enabled(),
                'config_summary': config_summary,
                'validation_details': []
            }

            if config.is_enabled():
                validation_result['validation_details'].append("Configuration is valid and enabled")
            else:
                validation_result['validation_details'].append("Configuration is disabled or has no valid rules")

            return validation_result

        except Exception as e:
            return {
                'is_valid': False,
                'error': f"Configuration validation failed: {e}",
                'validation_details': []
            }