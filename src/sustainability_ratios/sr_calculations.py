"""
Sustainability Ratios (SR) Calculations Module
==============================================

Main calculation engine for the SR module.
Integrates with existing BASIC calculations phase.

This module orchestrates:
- CSV panel configuration loading
- Market data retrieval
- Indicator calculations
- Chart generation
- Results saving
"""

import pandas as pd
import numpy as np
import os
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

from ..config import Config
from ..data_reader import DataReader
from ..user_defined_data import UserConfiguration
from .sr_config_reader import load_sr_configuration, parse_panel_config
from .sr_ratios import calculate_intermarket_ratios, calculate_market_breadth
from .sr_dashboard_generator import generate_sr_dashboard
from .sr_market_data import load_market_data_for_panels

logger = logging.getLogger(__name__)


def extract_tickers_from_data_source(data_source: str) -> List[str]:
    """
    Extract ticker symbols from various data source formats.

    Args:
        data_source: Data source string like "QQQ + EMA(QQQ, 10)", "SPY:QQQ", "A_PPO(12,26,9)_for_(QQQ)"

    Returns:
        List of ticker symbols
    """
    tickers = set()

    # Clean the data source
    clean_source = data_source.replace('"', '').strip()

    # Pattern 1: Tickers in _for_ clauses like "_for_(QQQ)" or "_for_(SPY,QQQ)"
    for_pattern = r'_for_\(([A-Z,\s]+)\)'
    for_matches = re.findall(for_pattern, clean_source)
    for match in for_matches:
        for ticker in match.split(','):
            ticker = ticker.strip()
            if ticker and len(ticker) <= 5 and ticker.isupper():
                tickers.add(ticker)

    # If no _for_ patterns, look for function parameters and simple tickers
    if not tickers:
        # Pattern 2: Tickers in functions like "EMA(QQQ, 10)", "PPO(12,26,9)"
        func_pattern = r'[A-Z]{1,5}(?=\s*[,)])'
        func_matches = re.findall(func_pattern, clean_source)
        for ticker in func_matches:
            if len(ticker) <= 5:
                tickers.add(ticker)

        # Pattern 3: Simple tickers like "QQQ", "SPY"
        simple_pattern = r'\b[A-Z]{1,5}\b'
        simple_matches = re.findall(simple_pattern, clean_source)
        for ticker in simple_matches:
            # Filter out common indicators
            if ticker not in ['EMA', 'SMA', 'PPO', 'RSI', 'PRICE', 'RATIO'] and len(ticker) <= 5:
                tickers.add(ticker)

    return sorted(list(tickers))


def get_latest_date_from_panels(panel_config: Dict, market_data: Dict[str, pd.DataFrame]) -> str:
    """
    Extract the latest available date from all tickers across all panels.

    Args:
        panel_config: Dictionary of panel configurations
        market_data: Dictionary of ticker DataFrames

    Returns:
        String formatted date (YYYYMMDD) or 'unknown'
    """
    latest_date = None
    all_tickers = set()

    logger.debug("Extracting tickers from panels for latest date calculation")

    for panel_name, panel_info in panel_config.items():
        data_source = panel_info.get('data_source', '')
        tickers = extract_tickers_from_data_source(data_source)
        logger.debug(f"Panel '{panel_name}': '{data_source}' → {tickers}")
        all_tickers.update(tickers)

    logger.debug(f"All unique tickers found: {sorted(list(all_tickers))}")

    for ticker in all_tickers:
        if ticker in market_data:
            data_max_date = market_data[ticker].index.max().date()
            logger.debug(f"Ticker {ticker}: latest date {data_max_date}")

            if latest_date is None or data_max_date > latest_date:
                latest_date = data_max_date
        else:
            logger.debug(f"Ticker {ticker}: no data available")

    if latest_date:
        formatted_date = latest_date.strftime('%Y%m%d')
        logger.info(f"Latest date found across all panels: {latest_date} → {formatted_date}")
        return formatted_date
    else:
        logger.warning("No valid dates found across panels → 'unknown'")
        return 'unknown'


def generate_sr_filename(file_id: str, row_number: int, user_choice: str, timeframe: str, latest_date: str) -> str:
    """
    Generate SR filename with improved structure.

    Args:
        file_id: File identifier from panel CSV
        row_number: Row number from panel processing
        user_choice: User choice from user_data.csv
        timeframe: Timeframe (daily/weekly/monthly)
        latest_date: Latest date string (YYYYMMDD)

    Returns:
        Generated filename
    """
    # Clean file_id (remove special characters, spaces)
    clean_file_id = re.sub(r'[^\w\-_]', '_', file_id)
    clean_file_id = re.sub(r'_+', '_', clean_file_id)  # Remove multiple underscores
    clean_file_id = clean_file_id.strip('_')  # Remove leading/trailing underscores

    # Format: sr_{file_id}_row{row_number}_{user_choice}_{timeframe}_{date}.png
    filename = f"sr_{clean_file_id}_row{row_number}_{user_choice}_{timeframe}_{latest_date}.png"

    logger.debug(f"Generated SR filename: {filename}")
    return filename


class SRProcessor:
    """
    Main processor for Sustainability Ratios analysis.
    """

    def __init__(self, config: Config, user_config: UserConfiguration, timeframe: str):
        """
        Initialize SR processor.

        Args:
            config: System configuration
            user_config: User configuration
            timeframe: Processing timeframe ('daily', 'weekly', 'monthly')
        """
        self.config = config
        self.user_config = user_config
        self.timeframe = timeframe
        self.data_reader = DataReader(config, timeframe, user_config.batch_size)
        self.sr_config = None
        self.panel_configs = []  # List of row configurations
        self.market_data = {}  # Global cache for all market data
        self.results = {}

        # 🔧 FIX: Load configuration during initialization
        self.load_configuration()

    def load_configuration(self) -> bool:
        """
        Load SR configuration from CSV files.

        Returns:
            True if configuration loaded successfully
        """
        try:
            # Load main SR configuration
            self.sr_config = load_sr_configuration(self.config, self.user_config)

            # Load panel configuration (now returns list of row configs)
            # 🔧 FIX: Use user configuration instead of hardcoded path
            panel_config_file = self.user_config.sr_panel_config_file
            if not os.path.isabs(panel_config_file):
                # Relative path - resolve from base directory
                panel_csv_path = Path(self.config.base_dir) / panel_config_file
            else:
                # Absolute path - use as is
                panel_csv_path = Path(panel_config_file)

            if panel_csv_path.exists():
                self.panel_configs = parse_panel_config(str(panel_csv_path))
                logger.info(f"Loaded SR panel configuration from {panel_csv_path} with {len(self.panel_configs)} row configurations")
                return True
            else:
                logger.warning(f"Panel configuration file not found: {panel_csv_path}")
                logger.info(f"Expected file: {panel_config_file} (from user config)")
                return False

        except Exception as e:
            logger.error(f"Error loading SR configuration: {e}")
            return False

    def load_all_market_data(self) -> bool:
        """
        Pre-load all market data needed for all panel configurations.

        Returns:
            True if market data loaded successfully
        """
        try:
            # Collect all unique tickers from all panel configs
            all_tickers = set()
            for panel_config in self.panel_configs:
                for panel_info in panel_config.values():
                    data_source = panel_info.get('data_source', '')
                    tickers = extract_tickers_from_data_source(data_source)
                    all_tickers.update(tickers)

            # Filter out invalid tickers
            valid_tickers = {ticker for ticker in all_tickers
                           if ticker and ticker != 'AUTO_DETECT' and len(ticker) <= 5}

            logger.info(f"Pre-loading market data for {len(valid_tickers)} unique tickers: {sorted(valid_tickers)}")

            # Load data for all tickers once
            loaded_count = 0
            for ticker in valid_tickers:
                try:
                    ticker_data = self.data_reader.read_stock_data(ticker)
                    if ticker_data is not None and not ticker_data.empty:
                        self.market_data[ticker] = ticker_data
                        loaded_count += 1
                except Exception as e:
                    logger.warning(f"Failed to load data for {ticker}: {e}")
                    continue

            logger.info(f"Successfully pre-loaded market data for {loaded_count}/{len(valid_tickers)} tickers")
            return loaded_count > 0

        except Exception as e:
            logger.error(f"Error loading market data: {e}")
            return False

    def _process_complex_data_sources(self, panel_config: Dict[str, Dict[str, Any]],
                                    raw_ticker_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Process complex data sources (bundled, ratio, enhanced formats) for a single row.

        Args:
            panel_config: Panel configuration for this row
            raw_ticker_data: Pre-loaded raw ticker data

        Returns:
            Dict with processed complex data sources
        """
        try:
            from .sr_market_data import decompose_data_source, process_decomposed_data_source

            complex_data = {}

            # Identify complex data sources that need processing
            for panel_name, panel_info in panel_config.items():
                data_source = panel_info.get('data_source', '')

                # Skip if already available in raw ticker data
                if data_source in raw_ticker_data:
                    continue

                # Check if this is a complex format that needs processing
                is_complex = (
                    '+' in data_source or  # Bundled format
                    ':' in data_source or  # Ratio format
                    '_for_(' in data_source or  # Enhanced format
                    panel_info.get('is_bundled', False) or
                    panel_info.get('is_multi_ticker', False)
                )

                if is_complex and data_source not in complex_data:
                    try:
                        # Decompose the data source
                        decomposition = decompose_data_source(data_source, panel_info)

                        # Process the decomposed data source
                        processed_data = process_decomposed_data_source(decomposition, raw_ticker_data)

                        if processed_data is not None:
                            complex_data[data_source] = processed_data
                            logger.debug(f"Processed complex data source: {data_source}")
                        else:
                            logger.warning(f"Failed to process complex data source: {data_source}")

                    except Exception as e:
                        logger.warning(f"Error processing complex data source '{data_source}': {e}")
                        continue

            logger.debug(f"Processed {len(complex_data)} complex data sources")
            return complex_data

        except Exception as e:
            logger.error(f"Error processing complex data sources: {e}")
            return {}

    def process_intermarket_ratios(self) -> Dict[str, pd.DataFrame]:
        """
        Calculate intermarket ratios for market timing analysis.
        DISABLED: Hardcoded ratios removed to prioritize panel configuration.

        Returns:
            Empty dict (function disabled)
        """
        logger.info("Intermarket ratios processing disabled - using panel configuration instead")
        return {}

    def process_market_breadth(self) -> Dict[str, Any]:
        """
        Calculate market breadth indicators.

        Returns:
            Dict with breadth calculations
        """
        try:
            logger.info("Calculating market breadth indicators...")

            # Get universe data for breadth calculation
            # Use basic tickers for breadth calculation
            universe_tickers = ['QQQ', 'SPY', 'XLY', 'XLP', 'IWF', 'IWD', 'XLK', 'XLF', 'XLV', 'XLE']

            breadth_data = {}
            for ticker in universe_tickers:  # Use basic ticker set
                try:
                    ticker_data = self.data_reader.read_stock_data(ticker)
                    if ticker_data is not None and not ticker_data.empty:
                        breadth_data[ticker] = ticker_data
                except Exception as e:
                    continue  # Skip problematic tickers

            if len(breadth_data) >= 10:
                breadth_results = calculate_market_breadth(breadth_data)
                self.results['market_breadth'] = breadth_results
                logger.info("Market breadth calculations completed")
                return breadth_results
            else:
                logger.warning("Insufficient data for market breadth calculations")
                return {}

        except Exception as e:
            logger.error(f"Error processing market breadth: {e}")
            return {}

    def process_panel_indicators(self) -> Dict[str, Any]:
        """
        Process indicators for each panel based on configuration.

        Returns:
            Dict with panel indicator results
        """
        try:
            # Combine all panel configs from all rows
            combined_panel_config = {}
            for row_config in self.panel_configs:
                combined_panel_config.update(row_config)

            if not combined_panel_config:
                logger.warning("No panel configuration available")
                return {}

            logger.info("Processing panel indicators...")

            # Load market data for all panels
            panel_data = load_market_data_for_panels(combined_panel_config, self.data_reader)

            # Process each panel (with or without indicators)
            panel_results = {}
            for panel_name, panel_info in combined_panel_config.items():
                if panel_info.get('data_source'):
                    try:
                        # Get data for this panel
                        data_source = panel_info['data_source']

                        # 🔧 SMART TYPE DETECTION: Route panels to appropriate processors
                        panel_type = self._detect_panel_type(data_source, panel_info)

                        if data_source in panel_data:
                            indicator = panel_info.get('indicator', '')

                            # 🔧 FIX: For legacy format, extract indicator from data_source if indicator field is empty
                            if not indicator and '_for_(' in data_source:
                                # Legacy format like "CORR(20)_for_(SPY,QQQ)" - extract "CORR"
                                import re
                                legacy_match = re.match(r'([A-Z]+)\(', data_source)
                                if legacy_match:
                                    indicator = legacy_match.group(1)

                            if indicator:
                                # 🔧 FIX: For data source indicators, create panel result from existing data
                                if panel_info.get('is_data_source_indicator', False):
                                    logger.debug(f"Creating panel result from existing data for data source indicator: {panel_name}")

                                    # Use existing data from panel_data[data_source]
                                    existing_data = panel_data[data_source]

                                    # Create panel result with existing data
                                    panel_result = {
                                        'data': existing_data,
                                        'name': panel_name,
                                        'data_source': data_source,
                                        'is_data_source_indicator': True
                                    }

                                    panel_results[panel_name] = {
                                        'data_source': data_source,
                                        'indicator': '',
                                        'result': panel_result,
                                        'timeframe': panel_info.get('timeframe', 'daily')
                                    }
                                    continue

                                # 🔧 TYPE-SPECIFIC INDICATOR PROCESSING

                                # Calculate indicator for this panel
                                from ..indicators.indicator_parser import calculate_indicator

                                # 🔧 FIX: Reconstruct full parameter string for indicator calculation
                                indicator_parameters = panel_info.get('indicator_parameters', {})

                                if indicator_parameters:
                                    # Build parameter string from parameters dict
                                    if indicator == 'PPO':
                                        param_string = f"PPO({indicator_parameters['fast_period']},{indicator_parameters['slow_period']},{indicator_parameters['signal_period']})"
                                    elif indicator == 'RSI':
                                        param_string = f"RSI({indicator_parameters['period']})"
                                    elif indicator in ['EMA', 'SMA']:
                                        param_string = f"{indicator}({indicator_parameters['period']})"
                                    elif indicator == 'RATIO':
                                        # For RATIO, tickers are stored separately in panel_info
                                        tickers = panel_info.get('tickers', indicator_parameters.get('tickers', []))
                                        if len(tickers) >= 2:
                                            param_string = f"RATIO({','.join(tickers)})"
                                        else:
                                            param_string = indicator
                                    else:
                                        # Generic fallback - try to build from available parameters
                                        param_values = list(indicator_parameters.values())
                                        if param_values:
                                            param_string = f"{indicator}({','.join(map(str, param_values))})"
                                        else:
                                            param_string = indicator
                                else:
                                    param_string = indicator


                                # 🔧 FIX: Special handling for correlation indicators that need multiple data sources
                                if indicator == 'CORR':
                                    # Check if this is pre-calculated correlation data (legacy format)
                                    if isinstance(panel_data[data_source], dict):
                                        indicator_result = panel_data[data_source]
                                    else:
                                        # Extract tickers from the data source (e.g., "B_CORR(20)_for_(SPY,QQQ)")
                                        tickers = extract_tickers_from_data_source(data_source)
                                        if len(tickers) >= 2:
                                            # Get data for both tickers
                                            ticker1, ticker2 = tickers[0], tickers[1]
                                            if ticker1 in panel_data and ticker2 in panel_data:
                                                from ..indicators.indicator_parser import calculate_indicator
                                                # Import CORR calculator directly since calculate_indicator doesn't support dual data
                                                from ..indicators.CORR import parse_corr_params, calculate_corr_for_chart

                                                # Parse parameters from the param_string
                                                corr_params = parse_corr_params(param_string)

                                                # Calculate correlation with both data sources
                                                indicator_result = calculate_corr_for_chart(
                                                    panel_data[ticker1],
                                                    panel_data[ticker2],
                                                    **corr_params
                                                )
                                            else:
                                                print(f"✗ Missing data for CORR calculation: {ticker1}={ticker1 in panel_data}, {ticker2}={ticker2 in panel_data}")
                                                indicator_result = None
                                        else:
                                            print(f"✗ CORR indicator needs 2 tickers, got: {tickers}")
                                            indicator_result = None
                                else:
                                    # Regular single-data-source indicators
                                    indicator_result = calculate_indicator(
                                        panel_data[data_source],
                                        param_string
                                    )

                                # Add stacking metadata to indicator results
                                if isinstance(indicator_result, dict) and 'metadata' in indicator_result:
                                    indicator_result['metadata'].update({
                                        'stacking_order': panel_info.get('stacking_order', 999),
                                        'position': panel_info.get('position', 'main'),
                                        'stacking_group': panel_info.get('stacking_group', '')
                                    })
                            else:
                                # No indicator - use raw price data
                                indicator_result = {
                                    'price': panel_data[data_source]['Close'],
                                    'metadata': {
                                        'chart_type': 'price',
                                        'data_type': 'ohlc',
                                        'stacking_order': panel_info.get('stacking_order', 999),
                                        'position': panel_info.get('position', 'main'),
                                        'stacking_group': panel_info.get('stacking_group', '')
                                    }
                                }

                            panel_results[panel_name] = {
                                'data_source': data_source,
                                'indicator': indicator,
                                'result': indicator_result,
                                'timeframe': panel_info.get('timeframe', 'daily')
                            }

                            if indicator:
                                logger.debug(f"Processed {panel_name}: {data_source} with {indicator}")
                            else:
                                logger.debug(f"Processed {panel_name}: {data_source} (price chart only)")

                    except Exception as e:
                        logger.warning(f"Error processing {panel_name}: {e}")
                        continue

            self.results['panel_indicators'] = panel_results
            logger.info(f"Processed indicators for {len(panel_results)} panels")
            return panel_results

        except Exception as e:
            logger.error(f"Error processing panel indicators: {e}")
            return {}

    def generate_charts(self) -> Dict[str, str]:
        """
        Generate SR dashboard charts.

        Returns:
            Dict with chart file paths
        """
        try:
            if not self.results:
                logger.warning("No results available for chart generation")
                return {}

            logger.info("Generating SR dashboard charts...")

            # Create output directory
            output_dir = Path(self.config.directories.get('SR_output_dir', 'results/sustainability_ratios'))
            output_dir.mkdir(parents=True, exist_ok=True)

            # Generate dashboard
            chart_paths = generate_sr_dashboard(
                self.results,
                output_dir,
                self.panel_config,
                self.user_config
            )

            self.results['chart_paths'] = chart_paths
            logger.info(f"Generated {len(chart_paths)} chart files")
            return chart_paths

        except Exception as e:
            logger.error(f"Error generating charts: {e}")
            return {}

    def save_results(self) -> bool:
        """
        Save SR analysis results to CSV files.

        Returns:
            True if saved successfully
        """
        try:
            # Create output directory
            output_dir = Path(self.config.directories.get('SR_output_dir', 'results/sustainability_ratios'))
            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d")

            # Save intermarket ratios
            if 'intermarket_ratios' in self.results:
                ratios_file = output_dir / f"intermarket_ratios_{timestamp}.csv"
                self.results['intermarket_ratios'].to_csv(ratios_file, index=True)
                logger.info(f"Saved intermarket ratios to {ratios_file}")

            # Save market breadth
            if 'market_breadth' in self.results:
                breadth_file = output_dir / f"market_breadth_{timestamp}.csv"
                if isinstance(self.results['market_breadth'], dict):
                    # Convert dict to DataFrame for saving
                    breadth_df = pd.DataFrame(self.results['market_breadth'])
                    breadth_df.to_csv(breadth_file, index=True)
                else:
                    self.results['market_breadth'].to_csv(breadth_file, index=True)
                logger.info(f"Saved market breadth to {breadth_file}")

            # Save panel results summary
            if 'panel_indicators' in self.results:
                summary_data = []
                for panel_name, panel_result in self.results['panel_indicators'].items():
                    summary_data.append({
                        'panel': panel_name,
                        'data_source': panel_result.get('data_source', ''),
                        'indicator': panel_result.get('indicator', ''),
                        'timeframe': panel_result.get('timeframe', ''),
                        'calculated': 'Yes' if panel_result.get('result') else 'No'
                    })

                if summary_data:
                    summary_df = pd.DataFrame(summary_data)
                    summary_file = output_dir / f"panel_summary_{timestamp}.csv"
                    summary_df.to_csv(summary_file, index=False)
                    logger.info(f"Saved panel summary to {summary_file}")

            return True

        except Exception as e:
            logger.error(f"Error saving SR results: {e}")
            return False

    def run_full_analysis(self) -> bool:
        """
        Run complete SR analysis pipeline.

        Returns:
            True if analysis completed successfully
        """
        try:
            logger.info("Starting Sustainability Ratios (SR) analysis...")

            # Step 1: Load configuration
            if not self.load_configuration():
                logger.error("Failed to load SR configuration")
                return False

            # Step 2: Process each row configuration separately
            if self.panel_configs:
                logger.info(f"Processing {len(self.panel_configs)} row configurations")
                self.process_all_row_configurations()
            else:
                logger.info("No panel configurations found - running default analysis")
                self.process_intermarket_ratios()
                self.process_market_breadth()
                self.generate_charts()

            # Step 3: Run Overview submodule (if enabled)
            if getattr(self.user_config, 'sr_overview_enable', False):
                logger.info("Running Overview submodule...")
                try:
                    from .overview import OverviewProcessor
                    overview_processor = OverviewProcessor(self.config, self.user_config, self.timeframe)
                    overview_results = overview_processor.run_complete_analysis()

                    if overview_results and overview_results.get('success', False):
                        logger.info(f"✅ Overview analysis completed: {len(overview_results.get('csv_files', []))} CSV files, {len(overview_results.get('chart_files', []))} charts")
                        self.results['overview_results'] = overview_results
                    else:
                        logger.warning("Overview analysis failed or returned no results")

                except Exception as e:
                    logger.error(f"Error in Overview submodule: {e}")

            # Step 4: Run MMM submodule (if enabled)
            if getattr(self.user_config, 'sr_mmm_enable', False):
                logger.info("Running MMM submodule...")
                try:
                    from .mmm import MmmProcessor
                    mmm_processor = MmmProcessor(self.config, self.user_config, self.timeframe)
                    mmm_results = mmm_processor.run_complete_analysis()

                    if mmm_results and mmm_results.get('success', False):
                        logger.info(f"✅ MMM analysis completed: {len(mmm_results.get('csv_files', []))} CSV files, {len(mmm_results.get('chart_files', []))} charts")
                        self.results['mmm_results'] = mmm_results
                    else:
                        logger.warning("MMM analysis failed or returned no results")

                except Exception as e:
                    logger.error(f"Error in MMM submodule: {e}")

            # Step 5: Save results
            self.save_results()

            logger.info("SR analysis completed successfully")
            return True

        except Exception as e:
            logger.error(f"Error in SR analysis: {e}")
            return False

    def process_all_row_configurations(self) -> bool:
        """
        Process each row configuration separately and generate individual charts.

        Returns:
            True if processing completed successfully
        """
        try:
            # Pre-load all market data once before processing
            if not self.load_all_market_data():
                logger.error("Failed to load market data - aborting row processing")
                return False

            self.results['row_results'] = []
            self.results['chart_paths'] = {}

            for row_idx, panel_config in enumerate(self.panel_configs):
                logger.info(f"Processing row configuration {row_idx + 1}/{len(self.panel_configs)}")

                # Process this row's panel indicators
                row_results = self.process_row_panel_indicators(panel_config, row_idx + 1)

                if row_results:
                    # Generate chart for this row
                    chart_path = self.generate_row_chart(row_results, row_idx + 1, panel_config)

                    # Store results
                    self.results['row_results'].append({
                        'row_number': row_idx + 1,
                        'panel_results': row_results,
                        'chart_path': chart_path
                    })

                    if chart_path:
                        self.results['chart_paths'][f'row_{row_idx + 1}'] = chart_path

            logger.info(f"Processed {len(self.results['row_results'])} row configurations")
            return True

        except Exception as e:
            logger.error(f"Error processing row configurations: {e}")
            return False

    def _get_file_id_from_config(self, panel_config: Dict[str, Dict[str, Any]], row_number: int) -> str:
        """
        Extract file_id from panel configuration or generate default.

        Args:
            panel_config: Panel configuration for this row
            row_number: Row number for fallback

        Returns:
            File identifier string
        """
        if panel_config:
            # Try to find file_name_id in configuration
            for panel_name, panel_info in panel_config.items():
                if 'file_name_id' in panel_info:
                    return panel_info['file_name_id']

                # Fallback: Use first panel's data_source as file_id
                if 'data_source' in panel_info:
                    data_source = panel_info['data_source']
                    # Extract meaningful name from data source
                    tickers = extract_tickers_from_data_source(data_source)
                    if tickers:
                        return '_'.join(tickers[:2])  # Use first 2 tickers
                    else:
                        return data_source.replace('"', '').replace(' ', '_')[:20]  # Truncate long names

        # Final fallback
        return f"Panel_Row_{row_number}"

    def _get_timeframe_from_config(self) -> str:
        """
        Get timeframe from configuration.

        Returns:
            Timeframe string (daily/weekly/monthly)
        """
        # Check user configuration for enabled timeframes
        if hasattr(self.user_config, 'sr_timeframe_daily') and self.user_config.sr_timeframe_daily:
            return 'daily'
        elif hasattr(self.user_config, 'sr_timeframe_weekly') and self.user_config.sr_timeframe_weekly:
            return 'weekly'
        elif hasattr(self.user_config, 'sr_timeframe_monthly') and self.user_config.sr_timeframe_monthly:
            return 'monthly'
        else:
            # Default fallback
            return 'daily'

    def _detect_panel_type(self, data_source: str, panel_info: Dict[str, Any]) -> str:
        """
        Smart type detection for panels based on data_source and configuration.

        Returns:
            'correlation' - CORR panels with _for_() format
            'complex' - Complex indicators with + operators
            'basic' - Simple ticker-based panels
        """
        # Type 1: Correlation panels (legacy format)
        if '_for_(' in data_source and data_source.endswith(')'):
            if 'CORR(' in data_source:
                return 'correlation'

        # Type 2: Complex indicators (bundled with operators)
        if '+' in data_source and any(op in data_source for op in ['SMA(', 'EMA(', 'INDEX_']):
            return 'complex'

        # Type 3: Basic panels (simple tickers, price charts)
        return 'basic'

    def process_row_panel_indicators(self, panel_config: Dict[str, Dict[str, Any]], row_number: int) -> Dict[str, Any]:
        """
        Process panel indicators for a single row configuration.

        Args:
            panel_config: Panel configuration for this row
            row_number: Row number for reference

        Returns:
            Dict with panel indicator results for this row
        """
        try:
            if not panel_config:
                logger.warning(f"No panel configuration available for row {row_number}")
                return {}

            logger.info(f"Processing panel indicators for row {row_number}...")


            # Load market data for this row using the same mechanism as main processing
            panel_data = load_market_data_for_panels(panel_config, self.data_reader)

            # Process each panel in this row
            panel_results = {}
            for panel_name, panel_info in panel_config.items():
                if panel_info.get('data_source'):
                    try:
                        # Get data for this panel
                        data_source = panel_info['data_source']

                        # 🔧 SMART TYPE DETECTION: Route panels to appropriate processors
                        panel_type = self._detect_panel_type(data_source, panel_info)

                        # 🔧 TYPE-SPECIFIC PROCESSING: Route to appropriate processor
                        if panel_type == 'correlation':
                            # Correlation panels should always be in panel_data after our fixes
                            if data_source not in panel_data:
                                continue

                        if data_source in panel_data:
                            indicator = panel_info.get('indicator', '')

                            # 🔧 FIX: For legacy format, extract indicator from data_source if indicator field is empty
                            if not indicator and '_for_(' in data_source:
                                # Legacy format like "CORR(20)_for_(SPY,QQQ)" - extract "CORR"
                                import re
                                legacy_match = re.match(r'([A-Z]+)\(', data_source)
                                if legacy_match:
                                    indicator = legacy_match.group(1)

                            if indicator:
                                # 🔧 FIX: For data source indicators, create panel result from existing data
                                if panel_info.get('is_data_source_indicator', False):
                                    logger.debug(f"Creating panel result from existing data for data source indicator: {panel_name}")

                                    # Use existing data from panel_data[data_source]
                                    existing_data = panel_data[data_source]

                                    # Create panel result with existing data
                                    panel_result = {
                                        'data': existing_data,
                                        'name': panel_name,
                                        'data_source': data_source,
                                        'is_data_source_indicator': True
                                    }

                                    panel_results[panel_name] = {
                                        'data_source': data_source,
                                        'indicator': '',
                                        'result': panel_result,
                                        'timeframe': panel_info.get('timeframe', 'daily')
                                    }
                                    continue

                                # 🔧 TYPE-SPECIFIC INDICATOR PROCESSING

                                # Calculate indicator for this panel
                                from ..indicators.indicator_parser import calculate_indicator

                                # 🔧 FIX: Reconstruct full parameter string for indicator calculation
                                indicator_parameters = panel_info.get('indicator_parameters', {})

                                if indicator_parameters:
                                    # Build parameter string from parameters dict
                                    if indicator == 'PPO':
                                        param_string = f"PPO({indicator_parameters['fast_period']},{indicator_parameters['slow_period']},{indicator_parameters['signal_period']})"
                                    elif indicator == 'RSI':
                                        param_string = f"RSI({indicator_parameters['period']})"
                                    elif indicator in ['EMA', 'SMA']:
                                        param_string = f"{indicator}({indicator_parameters['period']})"
                                    elif indicator == 'RATIO':
                                        # For RATIO, tickers are stored separately in panel_info
                                        tickers = panel_info.get('tickers', indicator_parameters.get('tickers', []))
                                        if len(tickers) >= 2:
                                            param_string = f"RATIO({','.join(tickers)})"
                                        else:
                                            param_string = indicator
                                    else:
                                        # Generic fallback - try to build from available parameters
                                        param_values = list(indicator_parameters.values())
                                        if param_values:
                                            param_string = f"{indicator}({','.join(map(str, param_values))})"
                                        else:
                                            param_string = indicator
                                else:
                                    param_string = indicator


                                # 🔧 FIX: Special handling for correlation indicators that need multiple data sources
                                if indicator == 'CORR':
                                    # Check if this is pre-calculated correlation data (legacy format)
                                    if isinstance(panel_data[data_source], dict):
                                        indicator_result = panel_data[data_source]
                                    else:
                                        # Extract tickers from the data source (e.g., "B_CORR(20)_for_(SPY,QQQ)")
                                        tickers = extract_tickers_from_data_source(data_source)
                                        if len(tickers) >= 2:
                                            # Get data for both tickers
                                            ticker1, ticker2 = tickers[0], tickers[1]
                                            if ticker1 in panel_data and ticker2 in panel_data:
                                                from ..indicators.indicator_parser import calculate_indicator
                                                # Import CORR calculator directly since calculate_indicator doesn't support dual data
                                                from ..indicators.CORR import parse_corr_params, calculate_corr_for_chart

                                                # Parse parameters from the param_string
                                                corr_params = parse_corr_params(param_string)

                                                # Calculate correlation with both data sources
                                                indicator_result = calculate_corr_for_chart(
                                                    panel_data[ticker1],
                                                    panel_data[ticker2],
                                                    **corr_params
                                                )
                                            else:
                                                print(f"✗ Missing data for CORR calculation: {ticker1}={ticker1 in panel_data}, {ticker2}={ticker2 in panel_data}")
                                                indicator_result = None
                                        else:
                                            print(f"✗ CORR indicator needs 2 tickers, got: {tickers}")
                                            indicator_result = None
                                else:
                                    # Regular single-data-source indicators
                                    indicator_result = calculate_indicator(
                                        panel_data[data_source],
                                        param_string
                                    )

                                # Add stacking metadata to indicator results
                                if isinstance(indicator_result, dict) and 'metadata' in indicator_result:
                                    indicator_result['metadata'].update({
                                        'stacking_order': panel_info.get('stacking_order', 999),
                                        'position': panel_info.get('position', 'main'),
                                        'stacking_group': panel_info.get('stacking_group', ''),
                                        'row_number': row_number
                                    })
                            else:
                                # No indicator specified - check if it's bundled format
                                is_bundled = '+' in data_source and any(x in data_source for x in ['EMA', 'SMA', 'PPO', 'RSI'])

                                if is_bundled:
                                    # 🎯 FIX: For bundled format, preserve all DataFrame columns (including EMA)
                                    df = panel_data[data_source]

                                    # Create result with all columns as Series
                                    indicator_result = {}
                                    for col in df.columns:
                                        if col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                                            # OHLCV data
                                            indicator_result[col] = df[col]
                                        elif col.startswith(('EMA_', 'SMA_', 'PPO_', 'RSI_')):
                                            # Indicator overlay data - preserve these!
                                            indicator_result[col] = df[col]
                                            # print(f"   Preserved indicator column: {col}")  # Debug output

                                    # Add metadata
                                    indicator_result['metadata'] = {
                                        'chart_type': 'overlay',  # bundled format should use overlay
                                        'data_type': 'bundled',
                                        'stacking_order': panel_info.get('stacking_order', 999),
                                        'position': panel_info.get('position', 'main'),
                                        'stacking_group': panel_info.get('stacking_group', ''),
                                        'row_number': row_number
                                    }
                                else:
                                    # Standard case - use raw price data
                                    indicator_result = {
                                        'price': panel_data[data_source]['Close'],
                                        'metadata': {
                                            'chart_type': 'price',
                                            'data_type': 'ohlc',
                                            'stacking_order': panel_info.get('stacking_order', 999),
                                            'position': panel_info.get('position', 'main'),
                                            'stacking_group': panel_info.get('stacking_group', ''),
                                            'row_number': row_number
                                        }
                                    }

                            # Determine if this is bundled format
                            is_bundled_format = '+' in data_source and any(x in data_source for x in ['EMA', 'SMA', 'PPO', 'RSI'])

                            panel_results[panel_name] = {
                                'data_source': data_source,
                                'indicator': indicator,
                                'result': indicator_result,
                                'timeframe': panel_info.get('timeframe', 'daily'),
                                'is_bundled': is_bundled_format  # Add bundled flag for chart generation
                            }

                            if indicator:
                                logger.debug(f"Processed {panel_name}: {data_source} with {indicator}")
                            else:
                                logger.debug(f"Processed {panel_name}: {data_source} (price chart only)")

                    except Exception as e:
                        logger.warning(f"Error processing {panel_name}: {e}")
                        continue

            logger.info(f"Processed indicators for {len(panel_results)} panels in row {row_number}")
            return panel_results

        except Exception as e:
            logger.error(f"Error processing panel indicators for row {row_number}: {e}")
            return {}

    def generate_row_chart(self, row_results: Dict[str, Any], row_number: int, panel_config: Dict[str, Dict[str, Any]] = None) -> str:
        """
        Generate chart for a single row configuration.

        Args:
            row_results: Panel results for this row
            row_number: Row number for filename
            panel_config: Panel configuration for this row (for extracting file_id and other metadata)

        Returns:
            Path to generated chart file
        """
        try:
            if not row_results:
                logger.warning(f"No results available for chart generation for row {row_number}")
                return ""

            logger.info(f"Generating chart for row {row_number}...")

            # Create output directory
            output_dir = Path(self.config.directories.get('SR_output_dir', 'results/sustainability_ratios'))
            output_dir.mkdir(parents=True, exist_ok=True)

            # Get configuration values for filename generation
            file_id = self._get_file_id_from_config(panel_config, row_number)
            user_choice = getattr(self.user_config, 'ticker_choice', '0')
            timeframe = self._get_timeframe_from_config()

            # Get latest date from market data
            latest_date = get_latest_date_from_panels(panel_config or {}, self.market_data)

            # Generate improved filename
            chart_filename = generate_sr_filename(file_id, row_number, user_choice, timeframe, latest_date)
            chart_path = output_dir / chart_filename

            # Use existing chart generation with row-specific title
            from .sr_dashboard_generator import create_multi_panel_chart
            chart_file = create_multi_panel_chart(
                row_results,
                str(chart_path),
                f"Sustainability Ratios Dashboard - Row {row_number}",
                self.user_config
            )

            if chart_file:
                logger.info(f"Generated chart for row {row_number}: {chart_file}")
                return chart_file
            else:
                logger.warning(f"Failed to generate chart for row {row_number}")
                return ""

        except Exception as e:
            logger.error(f"Error generating chart for row {row_number}: {e}")
            return ""


def run_sr_analysis(config: Config, user_config: UserConfiguration, timeframes: List[str]) -> Dict[str, Any]:
    """
    Main entry point for SR analysis following system patterns.

    Args:
        config: System configuration
        user_config: User configuration
        timeframes: List of timeframes to process

    Returns:
        Dict with analysis results for each timeframe
    """
    if not user_config.sr_enable:
        print(f"\n⏭️ Sustainability Ratios (SR) analysis disabled - skipping")
        return {}

    print(f"\n" + "="*60)
    print("📊 SUSTAINABILITY RATIOS (SR) ANALYSIS - ALL TIMEFRAMES")
    print("="*60)

    results_summary = {}
    total_processed = 0

    # Get SR-specific timeframes - new granular controls with backward compatibility
    active_timeframes = []

    # Check for new granular timeframe controls first
    if hasattr(user_config, 'sr_timeframe_daily') or hasattr(user_config, 'sr_timeframe_weekly') or hasattr(user_config, 'sr_timeframe_monthly'):
        # Use new granular controls
        if getattr(user_config, 'sr_timeframe_daily', True) and 'daily' in timeframes:
            active_timeframes.append('daily')
        if getattr(user_config, 'sr_timeframe_weekly', False) and 'weekly' in timeframes:
            active_timeframes.append('weekly')
        if getattr(user_config, 'sr_timeframe_monthly', False) and 'monthly' in timeframes:
            active_timeframes.append('monthly')

        logger.info(f"📊 Using new granular timeframe controls: daily={getattr(user_config, 'sr_timeframe_daily', True)}, "
                   f"weekly={getattr(user_config, 'sr_timeframe_weekly', False)}, "
                   f"monthly={getattr(user_config, 'sr_timeframe_monthly', False)}")
    else:
        # Fallback to legacy SR_timeframes setting
        sr_timeframes = user_config.sr_timeframes.split(';') if hasattr(user_config, 'sr_timeframes') else ['daily']
        active_timeframes = [tf for tf in timeframes if tf in sr_timeframes]
        logger.warning(f"📊 Using legacy SR_timeframes setting: {sr_timeframes}. Consider migrating to SR_timeframe_daily/weekly/monthly settings.")

    if not active_timeframes:
        logger.warning("No active timeframes for SR analysis")
        return {}

    for timeframe in active_timeframes:
        try:
            print(f"\n🔄 Processing SR analysis for {timeframe} timeframe...")

            # Create SR processor for this timeframe
            processor = SRProcessor(config, user_config, timeframe)
            success = processor.run_full_analysis()

            if success:
                # Count overview and MMM results
                overview_files = 0
                mmm_files = 0
                if 'overview_results' in processor.results:
                    overview_files = (len(processor.results['overview_results'].get('csv_files', [])) +
                                    len(processor.results['overview_results'].get('chart_files', [])))
                if 'mmm_results' in processor.results:
                    mmm_files = (len(processor.results['mmm_results'].get('csv_files', [])) +
                               len(processor.results['mmm_results'].get('chart_files', [])))

                results_summary[timeframe] = {
                    'status': 'completed',
                    'panels_processed': len(processor.results.get('panel_indicators', {})),
                    'charts_generated': len(processor.results.get('chart_paths', {})),
                    'overview_files': overview_files,
                    'mmm_files': mmm_files,
                    'overview_enabled': getattr(user_config, 'sr_overview_enable', False),
                    'mmm_enabled': getattr(user_config, 'sr_mmm_enable', False),
                    'timeframe': timeframe
                }
                total_processed += 1
                print(f"✅ {timeframe} SR analysis completed successfully")
            else:
                results_summary[timeframe] = {
                    'status': 'failed',
                    'panels_processed': 0,
                    'charts_generated': 0,
                    'timeframe': timeframe
                }
                print(f"⚠️ {timeframe} SR analysis completed with warnings")

        except Exception as e:
            logger.error(f"Error in SR analysis for {timeframe}: {e}")
            results_summary[timeframe] = {
                'status': 'error',
                'error': str(e),
                'timeframe': timeframe
            }
            print(f"❌ {timeframe} SR analysis failed: {e}")

    print(f"\n📊 SR ANALYSIS SUMMARY: {total_processed}/{len(active_timeframes)} timeframes completed")
    return results_summary