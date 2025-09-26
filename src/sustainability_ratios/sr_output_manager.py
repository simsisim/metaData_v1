"""
SR Output Manager
================

Centralized output directory management for sustainability ratios submodules
"""

from pathlib import Path
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class SROutputManager:
    """Manage output directories for sustainability ratios submodules"""

    # Define submodules and their output subdirectories
    SUBMODULES = {
        'overview': 'overview',           # Overview analysis charts and data
        'ratios': 'ratios',              # Intermarket ratios
        'breadth': 'breadth',            # Market breadth indicators
        'panels': 'panels',              # Panel-based dashboards (main dashboard output)
        'mmm': 'mmm',                    # Market Maker Manipulation analysis
        'calculations': 'calculations',   # Raw calculation outputs
        'config': 'config',              # Configuration files
        'reports': 'reports',            # Generated reports
        'debug': 'debug'                 # Debug outputs
    }

    def __init__(self, base_dir: str = "results/sustainability_ratios"):
        """
        Initialize output manager

        Args:
            base_dir: Base directory for all SR outputs
        """
        self.base_dir = Path(base_dir)
        self._ensure_directories()

    def _ensure_directories(self):
        """Create all required subdirectories"""
        try:
            # Create base directory
            self.base_dir.mkdir(parents=True, exist_ok=True)

            # Create submodule directories
            for submodule, subdir in self.SUBMODULES.items():
                submodule_path = self.base_dir / subdir
                submodule_path.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Ensured SR submodule directory: {submodule_path}")

        except Exception as e:
            logger.error(f"Error creating SR output directories: {e}")

    def get_submodule_dir(self, submodule: str) -> Path:
        """
        Get output directory for a specific submodule

        Args:
            submodule: Submodule name (from SUBMODULES keys)

        Returns:
            Path to submodule output directory
        """
        if submodule not in self.SUBMODULES:
            logger.warning(f"Unknown submodule '{submodule}', using base directory")
            return self.base_dir

        subdir = self.SUBMODULES[submodule]
        submodule_path = self.base_dir / subdir
        submodule_path.mkdir(parents=True, exist_ok=True)

        return submodule_path

    def get_output_path(self, submodule: str, filename: str) -> Path:
        """
        Get full output path for a file in a specific submodule

        Args:
            submodule: Submodule name
            filename: Output filename

        Returns:
            Full path for output file
        """
        submodule_dir = self.get_submodule_dir(submodule)
        return submodule_dir / filename

    def get_legacy_migration_map(self) -> Dict[str, str]:
        """
        Get mapping for migrating legacy files to new structure

        Returns:
            Dict mapping file patterns to submodules
        """
        return {
            'sr_overview_*.csv': 'overview',
            'sr_overview_*.png': 'overview',
            'sr_ratios_*.png': 'ratios',
            'sr_breadth_*.png': 'breadth',
            'sr_multi_panel_*.png': 'panels',
            'sr_*_row*_*.png': 'panels',  # Main dashboard charts
            'sr_calculations_*.csv': 'calculations',
            'sr_debug_*.txt': 'debug',
            'sr_report_*.pdf': 'reports'
        }

    def migrate_legacy_files(self):
        """Migrate existing files to new directory structure"""
        try:
            migration_map = self.get_legacy_migration_map()
            moved_files = 0

            # Look for files in base directory that should be in subdirectories
            for file_path in self.base_dir.glob('*'):
                if file_path.is_file():
                    filename = file_path.name
                    target_submodule = None

                    # Find matching pattern
                    for pattern, submodule in migration_map.items():
                        # Simple pattern matching (can be enhanced with fnmatch if needed)
                        pattern_clean = pattern.replace('*', '')
                        if pattern_clean in filename:
                            target_submodule = submodule
                            break

                    if target_submodule:
                        target_dir = self.get_submodule_dir(target_submodule)
                        target_path = target_dir / filename

                        # Move file if it doesn't already exist in target
                        if not target_path.exists():
                            file_path.rename(target_path)
                            moved_files += 1
                            logger.info(f"Migrated {filename} → {target_submodule}/")

            if moved_files > 0:
                logger.info(f"Migrated {moved_files} files to submodule directories")
            else:
                logger.debug("No legacy files found for migration")

        except Exception as e:
            logger.error(f"Error during legacy file migration: {e}")

    def cleanup_empty_legacy_dirs(self):
        """Remove empty directories after migration"""
        try:
            # This would be implemented if we had nested legacy structures
            pass
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def get_all_submodule_dirs(self) -> Dict[str, Path]:
        """Get all submodule directories"""
        return {
            submodule: self.get_submodule_dir(submodule)
            for submodule in self.SUBMODULES.keys()
        }

    def __str__(self) -> str:
        """String representation"""
        return f"SROutputManager(base_dir='{self.base_dir}')"


# Global instance for easy access
_default_manager: Optional[SROutputManager] = None


def get_sr_output_manager(base_dir: str = "results/sustainability_ratios") -> SROutputManager:
    """
    Get the default SR output manager instance

    Args:
        base_dir: Base directory for SR outputs

    Returns:
        SROutputManager instance
    """
    global _default_manager

    if _default_manager is None or str(_default_manager.base_dir) != base_dir:
        _default_manager = SROutputManager(base_dir)

    return _default_manager


# Convenience functions
def get_panels_output_dir(base_dir: str = "results/sustainability_ratios") -> Path:
    """Get panels submodule output directory"""
    return get_sr_output_manager(base_dir).get_submodule_dir('panels')


def get_overview_output_dir(base_dir: str = "results/sustainability_ratios") -> Path:
    """Get overview submodule output directory"""
    return get_sr_output_manager(base_dir).get_submodule_dir('overview')


def get_ratios_output_dir(base_dir: str = "results/sustainability_ratios") -> Path:
    """Get ratios submodule output directory"""
    return get_sr_output_manager(base_dir).get_submodule_dir('ratios')


def get_breadth_output_dir(base_dir: str = "results/sustainability_ratios") -> Path:
    """Get breadth submodule output directory"""
    return get_sr_output_manager(base_dir).get_submodule_dir('breadth')


if __name__ == "__main__":
    # Test the output manager
    logging.basicConfig(level=logging.INFO)

    manager = SROutputManager()
    print(f"SR Output Manager: {manager}")
    print(f"Submodule directories:")

    for submodule, path in manager.get_all_submodule_dirs().items():
        print(f"  {submodule}: {path}")

    # Test migration
    print(f"\nRunning legacy file migration...")
    manager.migrate_legacy_files()