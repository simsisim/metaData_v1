"""
Universe Generator Module
========================

Generates and manages ticker universe files for different market segments.
Creates organized universe files in results/ticker_universes/ directory.

Features:
- Auto-detection of missing universe files
- Smart filtering using boolean indicators
- Metadata tracking and validation
- Caching and regeneration logic
- Integration with existing configuration system
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import hashlib
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union

logger = logging.getLogger(__name__)


class UniverseGenerationError(Exception):
    """Custom exception for universe generation errors"""
    pass


class UniverseGenerator:
    """
    Main class for generating and managing ticker universe files.
    """
    
    # Universe definitions mapping
    UNIVERSE_DEFINITIONS = {
        '0': {
            'name': 'TradingView Universe',
            'description': 'All TradingView universe stocks',
            'filter_method': 'all',
            'source_file': 'tradingview_universe_bool.csv',
            'expected_count_range': (50, 150)
        },
        '1': {
            'name': 'S&P 500',
            'description': 'S&P 500 index members',
            'filter_method': 'boolean_column',
            'filter_column': 'SP500',
            'filter_value': True,
            'source_file': 'tradingview_universe_bool.csv',
            'expected_count_range': (400, 550)
        },
        '2': {
            'name': 'NASDAQ 100',
            'description': 'NASDAQ 100 index members',
            'filter_method': 'boolean_column',
            'filter_column': 'NASDAQ100',
            'filter_value': True,
            'source_file': 'tradingview_universe_bool.csv',
            'expected_count_range': (15, 110)
        },
        '3': {
            'name': 'NASDAQ Composite',
            'description': 'NASDAQ Composite index members',
            'filter_method': 'boolean_column',
            'filter_column': 'NASDAQComposite',
            'filter_value': True,
            'source_file': 'tradingview_universe_bool.csv',
            'expected_count_range': (20, 50)
        },
        '4': {
            'name': 'Russell 1000',
            'description': 'Russell 1000 index members',
            'filter_method': 'boolean_column',
            'filter_column': 'Russell1000',
            'filter_value': True,
            'source_file': 'tradingview_universe_bool.csv',
            'expected_count_range': (800, 1200)
        },
        '5': {
            'name': 'Index Tickers',
            'description': 'Index tickers only',
            'filter_method': 'external_file',
            'source_file': 'indexes_tickers.csv',
            'expected_count_range': (10, 50)
        }
    }
    
    def __init__(self, config, user_config):
        """
        Initialize universe generator.
        
        Args:
            config: Config object with directory paths
            user_config: User configuration object
        """
        self.config = config
        self.user_config = user_config
        
        # Create ticker_universes directory
        self.universes_dir = Path(config.directories['RESULTS_DIR']) / 'ticker_universes'
        self.universes_dir.mkdir(parents=True, exist_ok=True)
        
        # Metadata file path
        self.metadata_file = self.universes_dir / 'universe_metadata.json'
        
        logger.info(f"Universe Generator initialized: {self.universes_dir}")
    
    def detect_required_universes(self) -> List[str]:
        """
        Auto-detect which universes are needed based on configuration.
        
        Returns:
            List of universe IDs that are required
        """
        required = set()
        
        # Check RS percentile configuration
        if hasattr(self.user_config, 'rs_percentile_universe_stocks'):
            universes = self.user_config.rs_percentile_universe_stocks.split(';')
            required.update(u.strip() for u in universes if u.strip())
        
        # Add current ticker choice as required
        if hasattr(self.user_config, 'ticker_choice'):
            ticker_choice = str(self.user_config.ticker_choice).strip()
            # Handle combined choices like "0-5"
            if '-' in ticker_choice:
                for choice in ticker_choice.split('-'):
                    required.add(choice.strip())
            else:
                required.add(ticker_choice)
        
        # Filter to only supported universes
        supported_universes = set(self.UNIVERSE_DEFINITIONS.keys())
        required = required.intersection(supported_universes)
        
        logger.info(f"Detected required universes: {sorted(required)}")
        return sorted(required)
    
    def generate_missing_universes(self, required_universes: List[str]) -> Dict[str, Path]:
        """
        Generate only missing universe files.
        
        Args:
            required_universes: List of universe IDs to ensure exist
            
        Returns:
            Dict mapping universe_id to generated file path
        """
        results = {}
        
        for universe_id in required_universes:
            try:
                file_path = self._get_universe_file_path(universe_id)
                
                # Check if generation needed
                needs_regen, reason = self._is_regeneration_needed(universe_id)
                
                if needs_regen:
                    logger.info(f"Generating universe {universe_id}: {reason}")
                    results[universe_id] = self.generate_universe_file(universe_id)
                else:
                    logger.info(f"Universe {universe_id} is up to date: {file_path}")
                    results[universe_id] = file_path
                    
            except Exception as e:
                logger.error(f"Failed to process universe {universe_id}: {e}")
                # Continue with other universes
                
        return results
    
    def generate_universe_file(self, universe_id: str, force_regenerate: bool = False) -> Path:
        """
        Generate universe file with comprehensive error handling.
        
        Args:
            universe_id: Universe identifier
            force_regenerate: Force regeneration even if file exists
            
        Returns:
            Path to generated universe file
        """
        try:
            return self._generate_universe_primary(universe_id)
            
        except FileNotFoundError as e:
            logger.warning(f"Source file not found for universe {universe_id}: {e}")
            return self._handle_missing_source_file(universe_id)
            
        except KeyError as e:
            logger.warning(f"Required column not found for universe {universe_id}: {e}")
            return self._handle_missing_column(universe_id, str(e))
            
        except ValueError as e:
            logger.warning(f"Data validation failed for universe {universe_id}: {e}")
            return self._handle_validation_failure(universe_id, str(e))
            
        except Exception as e:
            logger.error(f"Unexpected error generating universe {universe_id}: {e}")
            return self._handle_fallback_generation(universe_id)
    
    def _generate_universe_primary(self, universe_id: str) -> Path:
        """
        Primary universe generation method.
        
        Args:
            universe_id: Universe identifier
            
        Returns:
            Path to generated universe file
        """
        universe_def = self.UNIVERSE_DEFINITIONS.get(universe_id)
        if not universe_def:
            raise ValueError(f"Unknown universe ID: {universe_id}")
        
        # Load source data
        source_df = self._load_source_data(universe_def['source_file'])
        
        # Apply filter
        filtered_df = self._apply_universe_filter(source_df, universe_id)
        
        # Validate result
        validation_result = self._validate_universe_data(universe_id, filtered_df)
        if validation_result['status'] != 'passed':
            logger.warning(f"Validation warnings for universe {universe_id}: {validation_result}")
        
        # Save universe file
        file_path = self._save_universe_file(filtered_df, universe_id)
        
        # Update metadata
        self._update_metadata(universe_id, file_path, len(filtered_df), validation_result)
        
        logger.info(f"Generated universe {universe_id}: {len(filtered_df)} stocks -> {file_path}")
        return file_path
    
    def _load_source_data(self, source_file: str) -> pd.DataFrame:
        """
        Load source data file.
        
        Args:
            source_file: Name of source file
            
        Returns:
            DataFrame with source data
        """
        # Try different possible locations
        possible_paths = [
            self.config.directories['TICKERS_DIR'] / source_file,
            Path('data/tickers') / source_file,
            Path(source_file)
        ]
        
        for file_path in possible_paths:
            if file_path.exists():
                logger.info(f"Loading source data from: {file_path}")
                return pd.read_csv(file_path)
        
        raise FileNotFoundError(f"Source file not found: {source_file}")
    
    def _apply_universe_filter(self, df: pd.DataFrame, universe_id: str) -> pd.DataFrame:
        """
        Apply universe-specific filter to DataFrame.
        
        Args:
            df: Source DataFrame
            universe_id: Universe identifier
            
        Returns:
            Filtered DataFrame
        """
        universe_def = self.UNIVERSE_DEFINITIONS[universe_id]
        filter_method = universe_def['filter_method']
        
        if filter_method == 'all':
            # No filtering, return all data
            return df.copy()
            
        elif filter_method == 'boolean_column':
            # Filter by boolean column
            filter_column = universe_def['filter_column']
            filter_value = universe_def['filter_value']
            
            if filter_column not in df.columns:
                raise KeyError(f"Filter column '{filter_column}' not found in source data")
            
            filtered_df = df[df[filter_column] == filter_value].copy()
            logger.info(f"Applied boolean filter {filter_column}=={filter_value}: {len(df)} -> {len(filtered_df)} stocks")
            return filtered_df
            
        elif filter_method == 'external_file':
            # Load from separate file (like indexes_tickers.csv)
            return self._load_external_universe_file(universe_def['source_file'])
            
        else:
            raise ValueError(f"Unknown filter method: {filter_method}")
    
    def _load_external_universe_file(self, source_file: str) -> pd.DataFrame:
        """
        Load universe data from external file.
        
        Args:
            source_file: Name of external file
            
        Returns:
            DataFrame with universe data
        """
        # Try to load the external file
        external_df = self._load_source_data(source_file)
        
        # If it's just a ticker list, create basic structure
        if 'ticker' not in external_df.columns and len(external_df.columns) == 1:
            ticker_col = external_df.columns[0]
            external_df = external_df.rename(columns={ticker_col: 'ticker'})
            external_df['description'] = f'Index ticker'
            external_df['source'] = 'external_file'
        
        return external_df
    
    def _get_universe_file_path(self, universe_id: str) -> Path:
        """
        Get the file path for a universe file.
        
        Args:
            universe_id: Universe identifier
            
        Returns:
            Path to universe file
        """
        filename = f'combined_info_tickers_clean_{universe_id}.csv'
        return self.universes_dir / filename
    
    def _save_universe_file(self, df: pd.DataFrame, universe_id: str) -> Path:
        """
        Save universe DataFrame to file.
        
        Args:
            df: Universe DataFrame
            universe_id: Universe identifier
            
        Returns:
            Path to saved file
        """
        file_path = self._get_universe_file_path(universe_id)
        
        # Add generation metadata to DataFrame
        df_to_save = df.copy()
        if 'generation_source' not in df_to_save.columns:
            df_to_save['generation_source'] = 'universe_generator'
        if 'generation_date' not in df_to_save.columns:
            df_to_save['generation_date'] = datetime.now().isoformat()
        
        # Save to file
        df_to_save.to_csv(file_path, index=False)
        logger.info(f"Saved universe file: {file_path} ({len(df_to_save)} records)")
        
        return file_path
    
    def _validate_universe_data(self, universe_id: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive validation of generated universe data.
        
        Args:
            universe_id: Universe identifier
            df: Universe DataFrame to validate
            
        Returns:
            Dict with validation results
        """
        validation_result = {
            'status': 'passed',
            'errors': [],
            'warnings': [],
            'details': {}
        }
        
        universe_def = self.UNIVERSE_DEFINITIONS[universe_id]
        
        # Record count validation
        expected_range = universe_def.get('expected_count_range', (1, 10000))
        actual_count = len(df)
        
        if actual_count < expected_range[0]:
            validation_result['errors'].append(f"Too few records: {actual_count} < {expected_range[0]}")
            validation_result['status'] = 'failed'
        elif actual_count > expected_range[1]:
            validation_result['warnings'].append(f"More records than expected: {actual_count} > {expected_range[1]}")
        
        validation_result['details']['record_count'] = actual_count
        validation_result['details']['expected_range'] = expected_range
        validation_result['details']['within_range'] = expected_range[0] <= actual_count <= expected_range[1]
        
        # Required columns validation
        required_columns = ['ticker']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_result['errors'].append(f"Missing required columns: {missing_columns}")
            validation_result['status'] = 'failed'
        
        # Data quality validation
        if 'ticker' in df.columns:
            null_tickers = df['ticker'].isnull().sum()
            if null_tickers > 0:
                validation_result['warnings'].append(f"Found {null_tickers} null tickers")
            
            duplicate_tickers = df['ticker'].duplicated().sum()
            if duplicate_tickers > 0:
                validation_result['errors'].append(f"Found {duplicate_tickers} duplicate tickers")
                validation_result['status'] = 'failed'
        
        return validation_result
    
    def _update_metadata(self, universe_id: str, file_path: Path, record_count: int, validation_result: Dict[str, Any]):
        """
        Update metadata for a generated universe file.
        
        Args:
            universe_id: Universe identifier
            file_path: Path to generated file
            record_count: Number of records in file
            validation_result: Validation results
        """
        metadata = self._load_metadata()
        
        # Calculate file hash for change detection
        file_hash = self._calculate_file_hash(file_path)
        
        # Create metadata entry
        metadata[universe_id] = {
            'generation_date': datetime.now().isoformat(),
            'universe_name': self.UNIVERSE_DEFINITIONS[universe_id]['name'],
            'source_file': self.UNIVERSE_DEFINITIONS[universe_id]['source_file'],
            'record_count': record_count,
            'file_path': str(file_path),
            'file_hash': file_hash,
            'file_size_bytes': file_path.stat().st_size,
            'validation_status': validation_result['status'],
            'validation_details': validation_result,
            'generation_method': 'primary'
        }
        
        # Save metadata
        self._save_metadata(metadata)
        logger.info(f"Updated metadata for universe {universe_id}")
    
    def _load_metadata(self) -> Dict[str, Any]:
        """
        Load existing metadata from file.
        
        Returns:
            Dict with metadata
        """
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")
        
        return {}
    
    def _save_metadata(self, metadata: Dict[str, Any]):
        """
        Save metadata to file.
        
        Args:
            metadata: Metadata dict to save
        """
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """
        Calculate MD5 hash of file for change detection.
        
        Args:
            file_path: Path to file
            
        Returns:
            MD5 hash string
        """
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.warning(f"Failed to calculate file hash: {e}")
            return "unknown"
    
    def _is_regeneration_needed(self, universe_id: str) -> Tuple[bool, str]:
        """
        Determine if regeneration is needed and why.
        
        Args:
            universe_id: Universe identifier
            
        Returns:
            (needs_regeneration: bool, reason: str)
        """
        file_path = self._get_universe_file_path(universe_id)
        
        # Check if file exists
        if not file_path.exists():
            return True, "File does not exist"
        
        # Load metadata
        metadata = self._load_metadata()
        universe_metadata = metadata.get(universe_id)
        
        if not universe_metadata:
            return True, "No metadata found"
        
        # Check if source file has changed
        universe_def = self.UNIVERSE_DEFINITIONS[universe_id]
        try:
            source_df = self._load_source_data(universe_def['source_file'])
            source_hash = self._calculate_dataframe_hash(source_df)
            
            if universe_metadata.get('source_hash') != source_hash:
                return True, "Source data has changed"
        except Exception:
            # If we can't check source, assume regeneration needed
            return True, "Cannot verify source data"
        
        # Check file integrity
        current_hash = self._calculate_file_hash(file_path)
        if universe_metadata.get('file_hash') != current_hash:
            return True, "File has been modified"
        
        # All checks passed
        return False, "Up to date"
    
    def _calculate_dataframe_hash(self, df: pd.DataFrame) -> str:
        """
        Calculate hash of DataFrame for change detection.
        
        Args:
            df: DataFrame to hash
            
        Returns:
            Hash string
        """
        try:
            # Convert DataFrame to string representation and hash
            df_string = df.to_string()
            return hashlib.md5(df_string.encode()).hexdigest()
        except Exception as e:
            logger.warning(f"Failed to calculate DataFrame hash: {e}")
            return "unknown"
    
    # Error handling and fallback methods
    def _handle_missing_source_file(self, universe_id: str) -> Path:
        """
        Fallback when primary source file is missing.
        
        Args:
            universe_id: Universe identifier
            
        Returns:
            Path to generated universe file
        """
        logger.warning(f"Handling missing source file for universe {universe_id}")
        
        # Try alternative source files
        alternative_sources = [
            'combined_info_tickers_clean_0.csv',
            'tradingview_universe.csv'
        ]
        
        for alt_source in alternative_sources:
            try:
                logger.info(f"Trying alternative source: {alt_source}")
                source_df = self._load_source_data(alt_source)
                
                # Apply filter if possible
                filtered_df = self._apply_universe_filter_fallback(source_df, universe_id)
                
                # Save with fallback metadata
                file_path = self._save_universe_file(filtered_df, universe_id)
                self._save_fallback_metadata(universe_id, file_path, 'alternative_source', alt_source)
                
                logger.info(f"Generated universe {universe_id} from alternative source: {alt_source}")
                return file_path
                
            except Exception as e:
                logger.warning(f"Alternative source {alt_source} failed: {e}")
                continue
        
        # Last resort: create minimal universe
        return self._create_minimal_universe(universe_id)
    
    def _handle_missing_column(self, universe_id: str, missing_column: str) -> Path:
        """
        Fallback when required boolean column is missing.
        
        Args:
            universe_id: Universe identifier
            missing_column: Name of missing column
            
        Returns:
            Path to generated universe file
        """
        logger.warning(f"Handling missing column {missing_column} for universe {universe_id}")
        
        # Try text-based filtering on 'index' column
        try:
            universe_def = self.UNIVERSE_DEFINITIONS[universe_id]
            source_df = self._load_source_data(universe_def['source_file'])
            
            if 'index' in source_df.columns:
                # Filter based on index name
                index_filter = universe_def['name'].upper()
                filtered_df = source_df[source_df['index'].str.contains(index_filter, na=False, case=False)].copy()
                
                if len(filtered_df) > 0:
                    file_path = self._save_universe_file(filtered_df, universe_id)
                    self._save_fallback_metadata(universe_id, file_path, 'text_filter', f"index contains '{index_filter}'")
                    
                    logger.info(f"Generated universe {universe_id} using text filter")
                    return file_path
        except Exception as e:
            logger.warning(f"Text filtering failed: {e}")
        
        # Fall back to external reference if available
        return self._handle_missing_source_file(universe_id)
    
    def _handle_validation_failure(self, universe_id: str, error_msg: str) -> Path:
        """
        Fallback when validation fails.
        
        Args:
            universe_id: Universe identifier
            error_msg: Validation error message
            
        Returns:
            Path to generated universe file
        """
        logger.warning(f"Handling validation failure for universe {universe_id}: {error_msg}")
        
        # For now, allow validation failures but mark them
        try:
            # Try to generate anyway with relaxed validation
            universe_def = self.UNIVERSE_DEFINITIONS[universe_id]
            source_df = self._load_source_data(universe_def['source_file'])
            filtered_df = self._apply_universe_filter(source_df, universe_id)
            
            # Save with warning metadata
            file_path = self._save_universe_file(filtered_df, universe_id)
            self._save_fallback_metadata(universe_id, file_path, 'validation_failed', error_msg)
            
            logger.warning(f"Generated universe {universe_id} despite validation failure")
            return file_path
            
        except Exception as e:
            logger.error(f"Relaxed generation also failed: {e}")
            return self._create_minimal_universe(universe_id)
    
    def _handle_fallback_generation(self, universe_id: str) -> Path:
        """
        Last resort fallback generation.
        
        Args:
            universe_id: Universe identifier
            
        Returns:
            Path to generated universe file
        """
        logger.error(f"Using last resort fallback for universe {universe_id}")
        return self._create_minimal_universe(universe_id)
    
    def _apply_universe_filter_fallback(self, df: pd.DataFrame, universe_id: str) -> pd.DataFrame:
        """
        Apply universe filter with fallback logic.
        
        Args:
            df: Source DataFrame
            universe_id: Universe identifier
            
        Returns:
            Filtered DataFrame
        """
        universe_def = self.UNIVERSE_DEFINITIONS[universe_id]
        
        # Try primary filter method
        try:
            return self._apply_universe_filter(df, universe_id)
        except Exception:
            pass
        
        # Fallback: try to filter by name in any text column
        universe_name = universe_def['name'].upper()
        text_columns = [col for col in df.columns if df[col].dtype == 'object']
        
        for col in text_columns:
            try:
                filtered_df = df[df[col].str.contains(universe_name, na=False, case=False)].copy()
                if len(filtered_df) > 0:
                    logger.info(f"Used fallback filter on column '{col}' for universe {universe_id}")
                    return filtered_df
            except Exception:
                continue
        
        # If all fails, return first 20 rows as minimal sample
        logger.warning(f"All filtering failed for universe {universe_id}, returning sample")
        return df.head(20).copy()
    
    def _create_minimal_universe(self, universe_id: str) -> Path:
        """
        Create minimal universe file as last resort.
        
        Args:
            universe_id: Universe identifier
            
        Returns:
            Path to generated universe file
        """
        logger.warning(f"Creating minimal universe for {universe_id}")
        
        # Get known tickers for this universe from external sources if possible
        known_tickers = self._get_known_tickers_for_universe(universe_id)
        
        # Create minimal DataFrame
        minimal_df = pd.DataFrame({
            'ticker': known_tickers,
            'description': f'Minimal {self.UNIVERSE_DEFINITIONS[universe_id]["name"]} universe',
            'source': 'fallback_generation',
            'generation_source': 'universe_generator_minimal',
            'generation_date': datetime.now().isoformat()
        })
        
        # Save with special metadata
        file_path = self._save_universe_file(minimal_df, universe_id)
        self._save_fallback_metadata(universe_id, file_path, 'minimal_generation', 'Created from known tickers')
        
        logger.warning(f"Created minimal universe file for {universe_id}: {file_path} ({len(minimal_df)} tickers)")
        return file_path
    
    def _get_known_tickers_for_universe(self, universe_id: str) -> List[str]:
        """
        Get known tickers for a universe as fallback.
        
        Args:
            universe_id: Universe identifier
            
        Returns:
            List of known ticker symbols
        """
        # Basic fallback tickers for each universe
        known_tickers = {
            '1': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],  # S&P 500 sample
            '2': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],  # NASDAQ 100 sample
            '3': ['AAPL', 'MSFT', 'GOOGL', 'AMZN'],          # NASDAQ Composite sample
            '4': ['AAPL', 'MSFT', 'GOOGL', 'AMZN'],          # Russell 1000 sample
            '5': ['SPY', 'QQQ', 'IWM', 'VTI'],               # Index tickers
        }
        
        return known_tickers.get(universe_id, ['SPY', 'QQQ'])
    
    def _save_fallback_metadata(self, universe_id: str, file_path: Path, method: str, details: str):
        """
        Save metadata for fallback generation.
        
        Args:
            universe_id: Universe identifier
            file_path: Path to generated file
            method: Fallback method used
            details: Details about fallback
        """
        metadata = self._load_metadata()
        
        metadata[universe_id] = {
            'generation_date': datetime.now().isoformat(),
            'universe_name': self.UNIVERSE_DEFINITIONS[universe_id]['name'],
            'record_count': self._count_file_lines(file_path) - 1,  # Subtract header
            'file_path': str(file_path),
            'file_hash': self._calculate_file_hash(file_path),
            'generation_method': 'fallback',
            'fallback_method': method,
            'fallback_details': details,
            'validation_status': 'warning'
        }
        
        self._save_metadata(metadata)
        logger.info(f"Saved fallback metadata for universe {universe_id}")
    
    def _count_file_lines(self, file_path: Path) -> int:
        """
        Count lines in a file.
        
        Args:
            file_path: Path to file
            
        Returns:
            Number of lines
        """
        try:
            with open(file_path, 'r') as f:
                return sum(1 for _ in f)
        except Exception:
            return 0