"""
PRE_PROCESS Configuration Loader
===============================

Loads and validates configuration from user_data_pre_process.csv.
Handles file type mapping, source/target paths, patterns, and filename templates.
"""

import pandas as pd
import os
from pathlib import Path
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class PreProcessConfig:
    """
    Configuration loader for PRE_PROCESS module.
    Reads user_data_pre_process.csv and validates settings.
    """

    def __init__(self, config_file: str = "user_data_pre_process.csv"):
        self.config_file = config_file
        self.config_data = None
        self.processing_rules = []
        self._load_config()

    def _load_config(self):
        """Load configuration from CSV file."""
        try:
            if not os.path.exists(self.config_file):
                raise FileNotFoundError(f"Configuration file not found: {self.config_file}")

            # Read CSV configuration
            self.config_data = pd.read_csv(self.config_file)
            logger.info(f"Loaded PRE_PROCESS configuration from {self.config_file}")

            # Parse processing rules
            self._parse_processing_rules()

        except Exception as e:
            logger.error(f"Failed to load PRE_PROCESS configuration: {e}")
            raise

    def _parse_processing_rules(self):
        """Parse processing rules from configuration data."""
        for _, row in self.config_data.iterrows():
            # Skip disabled rules
            if str(row['flag_file']).upper() != 'TRUE':
                continue

            rule = {
                'files_type': row['files_type'],
                'file_origin': row['file_origin'],
                'source_folder': row['files_path_folder_source'],
                'target_folder': row['files_path_folder_target'],
                'pattern_match': row['files_patterns_match'],
                'filename_template': row['new_filename']
            }

            # Validate rule
            if self._validate_rule(rule):
                self.processing_rules.append(rule)
                logger.info(f"Added processing rule: {rule['pattern_match']} -> {rule['target_folder']}")

    def _validate_rule(self, rule: Dict) -> bool:
        """Validate a single processing rule."""
        try:
            # Check required fields
            required_fields = ['files_type', 'file_origin', 'source_folder', 'target_folder',
                             'pattern_match', 'filename_template']
            for field in required_fields:
                if not rule.get(field):
                    logger.warning(f"Missing required field '{field}' in processing rule")
                    return False

            # Validate file type
            if rule['files_type'].lower() != 'csv':
                logger.warning(f"Unsupported file type: {rule['files_type']}")
                return False

            # Validate source directory exists
            source_path = Path(rule['source_folder'])
            if not source_path.exists():
                logger.warning(f"Source directory does not exist: {source_path}")
                return False

            # Create target directory if it doesn't exist
            target_path = Path(rule['target_folder'])
            target_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Ensured target directory exists: {target_path}")

            return True

        except Exception as e:
            logger.error(f"Error validating rule: {e}")
            return False

    def get_processing_rules(self) -> List[Dict]:
        """Get list of valid processing rules."""
        return self.processing_rules

    def get_source_folders(self) -> List[str]:
        """Get list of unique source folders."""
        return list(set(rule['source_folder'] for rule in self.processing_rules))

    def get_target_folders(self) -> List[str]:
        """Get list of unique target folders."""
        return list(set(rule['target_folder'] for rule in self.processing_rules))

    def get_patterns_for_folder(self, source_folder: str) -> List[str]:
        """Get pattern matches for a specific source folder."""
        patterns = []
        for rule in self.processing_rules:
            if rule['source_folder'] == source_folder:
                patterns.append(rule['pattern_match'])
        return patterns

    def get_rule_for_pattern(self, pattern: str) -> Optional[Dict]:
        """Get processing rule for a specific pattern."""
        for rule in self.processing_rules:
            if rule['pattern_match'] == pattern:
                return rule
        return None

    def is_enabled(self) -> bool:
        """Check if PRE_PROCESS is enabled (has valid rules)."""
        return len(self.processing_rules) > 0

    def get_filename_template(self, pattern: str) -> str:
        """Get filename template for a specific pattern."""
        rule = self.get_rule_for_pattern(pattern)
        return rule['filename_template'] if rule else '{ticker}.csv'

    def resolve_target_filename(self, ticker: str, pattern: str) -> str:
        """Resolve target filename using template and ticker."""
        template = self.get_filename_template(pattern)
        return template.replace('{ticker}', ticker)

    def get_file_origins(self) -> List[str]:
        """Get list of unique file origins."""
        return list(set(rule['file_origin'] for rule in self.processing_rules))

    def get_rules_by_origin(self, file_origin: str) -> List[Dict]:
        """Get processing rules for a specific file origin."""
        return [rule for rule in self.processing_rules if rule['file_origin'] == file_origin]

    def get_config_summary(self) -> Dict:
        """Get summary of configuration for logging."""
        return {
            'config_file': self.config_file,
            'total_rules': len(self.processing_rules),
            'file_origins': self.get_file_origins(),
            'source_folders': self.get_source_folders(),
            'target_folders': self.get_target_folders(),
            'enabled': self.is_enabled()
        }