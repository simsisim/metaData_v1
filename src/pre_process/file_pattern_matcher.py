"""
File Pattern Matcher
====================

Discovers files using glob patterns and maps them to target directories.
Handles batch processing and validates file accessibility.
"""

import os
import glob
from pathlib import Path
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class FilePatternMatcher:
    """
    Discovers files using pattern matching and maps source to target locations.
    """

    def __init__(self, config):
        self.config = config
        self.discovered_files = []

    def discover_files(self) -> List[Dict]:
        """
        Discover all files matching the configured patterns.

        Returns:
            List[Dict]: List of file mapping dictionaries
        """
        self.discovered_files = []

        for rule in self.config.get_processing_rules():
            source_folder = rule['source_folder']
            pattern = rule['pattern_match']
            target_folder = rule['target_folder']

            # Find files matching pattern in source folder
            files = self._find_files_by_pattern(source_folder, pattern)

            for file_path in files:
                file_info = {
                    'source_path': file_path,
                    'source_folder': source_folder,
                    'target_folder': target_folder,
                    'pattern': pattern,
                    'filename': os.path.basename(file_path),
                    'rule': rule
                }
                self.discovered_files.append(file_info)

        logger.info(f"Discovered {len(self.discovered_files)} files for processing")
        return self.discovered_files

    def _find_files_by_pattern(self, source_folder: str, pattern: str) -> List[str]:
        """
        Find files matching a specific pattern in source folder.

        Args:
            source_folder: Source directory path
            pattern: Pattern to match (e.g., '*1D*', '*1W*')

        Returns:
            List[str]: List of matching file paths
        """
        try:
            # Construct search pattern
            search_pattern = os.path.join(source_folder, f"*{pattern}*.csv")

            # Find matching files
            matching_files = glob.glob(search_pattern)

            # Validate files
            valid_files = []
            for file_path in matching_files:
                if self._validate_file(file_path):
                    valid_files.append(file_path)

            logger.debug(f"Found {len(valid_files)} valid files for pattern '{pattern}' in {source_folder}")
            return valid_files

        except Exception as e:
            logger.error(f"Error finding files with pattern '{pattern}' in {source_folder}: {e}")
            return []

    def _validate_file(self, file_path: str) -> bool:
        """
        Validate that a file exists and is accessible.

        Args:
            file_path: Path to file

        Returns:
            bool: True if file is valid
        """
        try:
            # Check file exists
            if not os.path.exists(file_path):
                logger.warning(f"File does not exist: {file_path}")
                return False

            # Check file is readable
            if not os.access(file_path, os.R_OK):
                logger.warning(f"File is not readable: {file_path}")
                return False

            # Check file is not empty
            if os.path.getsize(file_path) == 0:
                logger.warning(f"File is empty: {file_path}")
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating file {file_path}: {e}")
            return False

    def get_files_by_pattern(self, pattern: str) -> List[Dict]:
        """
        Get discovered files for a specific pattern.

        Args:
            pattern: Pattern to filter by

        Returns:
            List[Dict]: List of file info dictionaries
        """
        return [f for f in self.discovered_files if f['pattern'] == pattern]

    def get_files_by_target_folder(self, target_folder: str) -> List[Dict]:
        """
        Get discovered files for a specific target folder.

        Args:
            target_folder: Target folder to filter by

        Returns:
            List[Dict]: List of file info dictionaries
        """
        return [f for f in self.discovered_files if f['target_folder'] == target_folder]

    def get_unique_patterns(self) -> List[str]:
        """Get list of unique patterns from discovered files."""
        return list(set(f['pattern'] for f in self.discovered_files))

    def get_unique_target_folders(self) -> List[str]:
        """Get list of unique target folders from discovered files."""
        return list(set(f['target_folder'] for f in self.discovered_files))

    def get_file_count_by_pattern(self) -> Dict[str, int]:
        """Get count of files by pattern."""
        counts = {}
        for file_info in self.discovered_files:
            pattern = file_info['pattern']
            counts[pattern] = counts.get(pattern, 0) + 1
        return counts

    def get_processing_summary(self) -> Dict:
        """Get summary of file discovery for logging."""
        return {
            'total_files': len(self.discovered_files),
            'unique_patterns': self.get_unique_patterns(),
            'unique_targets': self.get_unique_target_folders(),
            'file_counts_by_pattern': self.get_file_count_by_pattern()
        }

    def prepare_batch_processing(self, batch_size: int = 10) -> List[List[Dict]]:
        """
        Prepare files for batch processing.

        Args:
            batch_size: Number of files per batch

        Returns:
            List[List[Dict]]: List of file batches
        """
        batches = []
        for i in range(0, len(self.discovered_files), batch_size):
            batch = self.discovered_files[i:i + batch_size]
            batches.append(batch)

        logger.info(f"Prepared {len(batches)} batches for processing (batch_size={batch_size})")
        return batches