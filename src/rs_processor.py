"""
Relative Strength Processing Engine
==================================

Main processor that orchestrates RS calculations for stocks, sectors, and industries.
Integrates with the existing post-processing pipeline and handles batch operations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

from .rs_ibd import IBDRelativeStrengthCalculator
from .sector_composite import SectorCompositeBuilder

# Import MA calculator only if needed
try:
    from .rs_ma import MARelativeStrengthCalculator
    MA_AVAILABLE = True
except ImportError:
    MA_AVAILABLE = False
    logger.warning("MA RS calculator not available - rs_ma module not found")

logger = logging.getLogger(__name__)


class RSProcessor:
    """
    Main processor for Relative Strength calculations across the entire universe.
    """
    
    def __init__(self, config, user_config):
        """
        Initialize RS processor.
        
        Args:
            config: Config object with directory paths
            user_config: User configuration object with RS settings
        """
        self.config = config
        self.user_config = user_config
        self.ibd_calculator = IBDRelativeStrengthCalculator(config, user_config)
        self.composite_builder = SectorCompositeBuilder(config, user_config)

        # Initialize MA calculator if enabled and available
        self.ma_calculator = None
        if MA_AVAILABLE and getattr(user_config, 'rs_ma_enable', False):
            try:
                self.ma_calculator = MARelativeStrengthCalculator(config, user_config)
                logger.info("MA RS calculator initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize MA RS calculator: {e}")
                self.ma_calculator = None
        
    def process_rs_analysis(self, ticker_list, ticker_choice=0, timeframes=None):
        """
        Run complete RS analysis for stocks, sectors, and industries.

        Args:
            ticker_list: List of ticker symbols to analyze
            ticker_choice: User ticker choice number
            timeframes: List of timeframes to process (overrides individual RS flags if provided)

        Returns:
            Dictionary with results summary
        """
        logger.info(f"Starting RS analysis for {len(ticker_list)} tickers (choice {ticker_choice})")
        
        # Support multiple benchmarks from RS_benchmark_tickers config
        benchmark_tickers = self._parse_benchmark_tickers()
        logger.info(f"Using benchmarks: {benchmark_tickers}")
        
        results_summary = {
            'ticker_choice': ticker_choice,
            'universe_size': len(ticker_list),
            'benchmark_ticker': ';'.join(benchmark_tickers),  # Show all benchmarks
            'timeframes_processed': [],
            'files_created': [],
            'errors': []
        }
        
        # Use provided timeframes (from main.py YF_*_data flags) or fallback to RS-specific flags
        if timeframes is None:
            # Legacy behavior - use RS-specific config flags
            timeframes = []
            if getattr(self.user_config, 'rs_daily_enable', True):
                timeframes.append('daily')
            if getattr(self.user_config, 'rs_weekly_enable', False):
                timeframes.append('weekly')
            if getattr(self.user_config, 'rs_monthly_enable', False):
                timeframes.append('monthly')

        logger.info(f"Processing RS for available timeframes: {timeframes}")

        # Process each timeframe - follow same pattern as basic_calculation and stage_analysis
        for timeframe in timeframes:
            # Check RS enable flag for this timeframe (hierarchical flag logic)
            rs_enabled = getattr(self.user_config, f'rs_{timeframe}_enable', True)
            if not rs_enabled:
                print(f"⏭️  RS analysis disabled for {timeframe} timeframe")
                continue

            print(f"\n📊 Processing RS {timeframe.upper()} timeframe...")
            try:
                timeframe_results = self._process_timeframe(ticker_list, timeframe, ticker_choice, benchmark_tickers)
                results_summary['timeframes_processed'].append(timeframe)
                results_summary['files_created'].extend(timeframe_results['files_created'])
                
            except Exception as e:
                error_msg = f"Error processing {timeframe}: {e}"
                logger.error(error_msg)
                results_summary['errors'].append(error_msg)
        
        logger.info(f"RS analysis completed. Files created: {len(results_summary['files_created'])}")
        return results_summary
    
    def _parse_benchmark_tickers(self):
        """Parse benchmark tickers from configuration, supporting multiple benchmarks."""
        # Try RS_benchmark_tickers (multiple) first
        if hasattr(self.user_config, 'rs_benchmark_tickers'):
            tickers_str = str(self.user_config.rs_benchmark_tickers).strip()
            if tickers_str:
                return [t.strip() for t in tickers_str.split(';') if t.strip()]
        
        # Fallback to legacy rs_benchmark_ticker (single)
        if hasattr(self.user_config, 'rs_benchmark_ticker'):
            return [self.user_config.rs_benchmark_ticker]
        
        # Default fallback
        return ['SPY']
    
    def _process_timeframe(self, ticker_list, timeframe, ticker_choice, benchmark_tickers):
        """
        Process RS analysis for a specific timeframe with multiple benchmarks.

        Args:
            ticker_list: List of tickers to analyze
            timeframe: Data timeframe ('daily', 'weekly', 'monthly')
            ticker_choice: User ticker choice number
            benchmark_tickers: List of benchmark tickers

        Returns:
            Dictionary with timeframe results
        """
        logger.info(f"Starting {timeframe} RS analysis for {len(ticker_list)} tickers")

        # Get batch size for actual RS processing (not display simulation)
        batch_size = getattr(self.user_config, 'batch_size', 100)  # Use standard batch_size from user config
        total_tickers = len(ticker_list)
        import math
        total_batches = math.ceil(total_tickers / batch_size) if batch_size < total_tickers else 1

        logger.info(f"Processing {total_tickers} tickers in {total_batches} batches of {batch_size}")

        timeframe_results = {
            'timeframe': timeframe,
            'files_created': [],
            'universe_info': None,
            'composite_indices': None
        }

        # 1. STOCK-LEVEL RS ANALYSIS - SEPARATE FILES PER BENCHMARK
        if getattr(self.user_config, 'rs_enable_stocks', True):
            # Process each benchmark separately and save individual files (like MA method)
            for benchmark_ticker in benchmark_tickers:
                try:
                    logger.info(f"Processing RS vs {benchmark_ticker} benchmark")
                    stock_results = self.ibd_calculator.process_universe(
                        ticker_list, timeframe, benchmark_ticker, batch_size
                    )

                    if stock_results.rs_values:
                        # Save separate file for this benchmark
                        stock_files = self.ibd_calculator.save_rs_results(
                            stock_results, 'stocks', str(ticker_choice), benchmark_ticker
                        )
                        timeframe_results['files_created'].extend(stock_files)
                        logger.info(f"Stock RS vs {benchmark_ticker} completed: {len(stock_files)} files created")
                    else:
                        logger.warning(f"No stock RS results generated for {benchmark_ticker}")

                except Exception as e:
                    logger.error(f"Error in stock RS analysis vs {benchmark_ticker}: {e}")

        # 1B. MA STOCK-LEVEL RS ANALYSIS (PARALLEL TO IBD)
        if self.ma_calculator and getattr(self.user_config, 'rs_enable_stocks', True):
            try:
                logger.info(f"Processing MA RS for stocks")
                ma_stock_results = self.ma_calculator.process_stocks(
                    ticker_list, timeframe, ticker_choice, benchmark_tickers
                )

                if ma_stock_results and 'error' not in ma_stock_results.metadata:
                    files_created = ma_stock_results.metadata.get('files_created', [])
                    timeframe_results['files_created'].extend(files_created)
                    logger.info(f"MA stock RS completed: {len(files_created)} files created")
                else:
                    error_msg = ma_stock_results.metadata.get('error', 'Unknown error') if ma_stock_results else 'No results'
                    logger.warning(f"MA stock RS failed: {error_msg}")

            except Exception as e:
                logger.error(f"Error in MA stock RS analysis: {e}")

        # 2. SECTOR/INDUSTRY COMPOSITE ANALYSIS - COMBINED ALL BENCHMARKS
        if (getattr(self.user_config, 'rs_enable_sectors', False) or 
            getattr(self.user_config, 'rs_enable_industries', False)):
            logger.info("Building sector/industry composites...")
            try:
                # Load universe information with sector/industry classifications
                universe_info = self.composite_builder.load_universe_info(ticker_choice)
                timeframe_results['universe_info'] = universe_info
                
                if not universe_info.empty:
                    # Load price data for composite building
                    price_data = self.ibd_calculator._load_price_data(ticker_list, timeframe)
                    
                    if not price_data.empty:
                        # Build composite indices
                        composite_indices = self.composite_builder.build_composite_indices(
                            price_data, getattr(self.user_config, 'rs_composite_method', 'equal_weighted')
                        )
                        timeframe_results['composite_indices'] = composite_indices
                        
                        # Save composite indices
                        composite_files = self.composite_builder.save_composite_indices(
                            composite_indices, ticker_choice, timeframe
                        )
                        timeframe_results['files_created'].extend(composite_files.values())
                        
                        # 3. SECTOR RS ANALYSIS - ALL BENCHMARKS COMBINED
                        if (getattr(self.user_config, 'rs_enable_sectors', False) and 
                            composite_indices['sectors']):
                            logger.info(f"Running sector RS analysis for all benchmarks: {benchmark_tickers}...")
                            sector_results = self._process_composite_rs_combined(
                                composite_indices['sectors'], timeframe, 'sectors', 
                                ticker_choice, benchmark_tickers
                            )
                            if sector_results:
                                timeframe_results['files_created'].extend(sector_results)
                        
                        # 4. INDUSTRY RS ANALYSIS - ALL BENCHMARKS COMBINED
                        if (getattr(self.user_config, 'rs_enable_industries', False) and 
                            composite_indices['industries']):
                            logger.info(f"Running industry RS analysis for all benchmarks: {benchmark_tickers}...")  
                            industry_results = self._process_composite_rs_combined(
                                composite_indices['industries'], timeframe, 'industries', 
                                ticker_choice, benchmark_tickers
                            )
                            if industry_results:
                                timeframe_results['files_created'].extend(industry_results)

                        # 3B. MA SECTOR RS ANALYSIS (PARALLEL TO IBD)
                        if (self.ma_calculator and getattr(self.user_config, 'rs_enable_sectors', False) and
                            composite_indices['sectors']):
                            try:
                                logger.info(f"Processing MA RS for sectors")
                                # Convert dict format to DataFrame format for MA calculator
                                sector_composites_df = self._convert_composites_to_dataframe(composite_indices['sectors'])
                                ma_sector_results = self.ma_calculator.process_sectors(
                                    sector_composites_df, timeframe, ticker_choice, benchmark_tickers
                                )

                                if ma_sector_results and 'error' not in ma_sector_results.metadata:
                                    files_created = ma_sector_results.metadata.get('files_created', [])
                                    timeframe_results['files_created'].extend(files_created)
                                    logger.info(f"MA sector RS completed: {len(files_created)} files created")
                                else:
                                    error_msg = ma_sector_results.metadata.get('error', 'Unknown error') if ma_sector_results else 'No results'
                                    logger.warning(f"MA sector RS failed: {error_msg}")

                            except Exception as e:
                                logger.error(f"Error in MA sector RS analysis: {e}")

                        # 4B. MA INDUSTRY RS ANALYSIS (PARALLEL TO IBD)
                        if (self.ma_calculator and getattr(self.user_config, 'rs_enable_industries', False) and
                            composite_indices['industries']):
                            try:
                                logger.info(f"Processing MA RS for industries")
                                # Convert dict format to DataFrame format for MA calculator
                                industry_composites_df = self._convert_composites_to_dataframe(composite_indices['industries'])
                                ma_industry_results = self.ma_calculator.process_industries(
                                    industry_composites_df, timeframe, ticker_choice, benchmark_tickers
                                )

                                if ma_industry_results and 'error' not in ma_industry_results.metadata:
                                    files_created = ma_industry_results.metadata.get('files_created', [])
                                    timeframe_results['files_created'].extend(files_created)
                                    logger.info(f"MA industry RS completed: {len(files_created)} files created")
                                else:
                                    error_msg = ma_industry_results.metadata.get('error', 'Unknown error') if ma_industry_results else 'No results'
                                    logger.warning(f"MA industry RS failed: {error_msg}")

                            except Exception as e:
                                logger.error(f"Error in MA industry RS analysis: {e}")

                    else:
                        logger.warning(f"No price data available for {timeframe} composite analysis")
                else:
                    logger.warning("No universe info available for composite analysis")
                    
            except Exception as e:
                logger.error(f"Error in sector/industry analysis: {e}")
        
        # Show completion summary like streaming implementations
        tickers_processed = len([f for f in timeframe_results['files_created'] if 'stocks' in str(f)])
        if tickers_processed > 0:
            logger.info(f"RS analysis completed for {timeframe}: {total_tickers} tickers processed")
            logger.info(f"Results saved to: {timeframe_results['files_created'][0] if timeframe_results['files_created'] else 'No files created'}")

        logger.info(f"{timeframe} RS analysis completed")
        return timeframe_results
    
    def _process_composite_rs(self, composite_indices, timeframe, level, ticker_choice, benchmark_ticker):
        """
        Process RS analysis for composite indices (sectors or industries).
        
        Args:
            composite_indices: Dictionary of {group_name: price_series}
            timeframe: Data timeframe
            level: Analysis level ('sectors' or 'industries')
            ticker_choice: User ticker choice number
            benchmark_ticker: Benchmark ticker to use
            
        Returns:
            List of created file paths
        """
        try:
            # Get analysis periods for this timeframe  
            periods = self.ibd_calculator.get_analysis_periods(timeframe)
            
            # Load benchmark data
            benchmark_data = self.ibd_calculator.load_benchmark_data(benchmark_ticker, timeframe)
            
            if benchmark_data is None:
                logger.warning(f"No benchmark data for {level} RS analysis vs {benchmark_ticker}")
                return []
            
            # Calculate RS for composite indices
            rs_data = self.ibd_calculator.calculate_composite_rs(
                composite_indices, benchmark_data, periods, benchmark_ticker
            )
            
            if not rs_data:
                logger.warning(f"No RS data calculated for {level}")
                return []
            
            # Create results container and save files
            from .rs_base import RSResults
            results = RSResults('ibd', level, timeframe)
            results.metadata['benchmark_ticker'] = benchmark_ticker
            results.metadata['universe_size'] = len(composite_indices)
            
            for column_suffix, rs_df in rs_data.items():
                if not rs_df.empty:
                    # Calculate percentile rankings for the RS column using new naming convention
                    rs_col = f'{column_suffix}_rs_vs_{benchmark_ticker}'
                    if rs_col in rs_df.columns:
                        percentiles = self.ibd_calculator._calculate_percentile_rankings(rs_df[rs_col])
                        rs_df[f'{column_suffix}_rs_percentile'] = percentiles
                    
                    results.rs_values[column_suffix] = rs_df
            
            # Save results
            saved_files = self.ibd_calculator.save_rs_results(results, level, str(ticker_choice), benchmark_ticker)
            
            logger.info(f"{level.capitalize()} RS analysis vs {benchmark_ticker} completed: {len(saved_files)} files created")
            return saved_files
            
        except Exception as e:
            logger.error(f"Error processing {level} RS analysis vs {benchmark_ticker}: {e}")
            return []
    
    def get_rs_summary(self, ticker_choice=0):
        """
        Generate summary statistics for RS analysis.
        
        Args:
            ticker_choice: User ticker choice number
            
        Returns:
            Dictionary with summary statistics
        """
        benchmark_tickers = self._parse_benchmark_tickers()
        
        summary = {
            'configuration': {
                'stocks_enabled': getattr(self.user_config, 'rs_enable_stocks', True),
                'sectors_enabled': getattr(self.user_config, 'rs_enable_sectors', False),
                'industries_enabled': getattr(self.user_config, 'rs_enable_industries', False),
                'benchmark_tickers': benchmark_tickers,
                'composite_method': getattr(self.user_config, 'rs_composite_method', 'equal_weighted'),
                'daily_periods': getattr(self.user_config, 'rs_daily_periods', [1, 3, 5, 10, 15]),
                'weekly_periods': getattr(self.user_config, 'rs_weekly_periods', [1, 2, 4, 6]),
                'monthly_periods': getattr(self.user_config, 'rs_monthly_periods', [1, 3, 6])
            },
            'group_statistics': {},
            'output_directory': self.config.directories['RESULTS_DIR'] / 'rs'
        }
        
        # Set group statistics
        if (getattr(self.user_config, 'rs_enable_sectors', False) or 
            getattr(self.user_config, 'rs_enable_industries', False)):
            summary['group_statistics'] = {
                'sector_count': 0,
                'industry_count': 0
            }
        
        return summary
    
    def _combine_benchmark_results(self, all_benchmark_data):
        """
        Combine RS results from multiple benchmarks into single DataFrames per period.
        
        Args:
            all_benchmark_data: Dict of {benchmark: {column_suffix: DataFrame}}
            
        Returns:
            Dict of {column_suffix: combined_DataFrame}
        """
        combined_results = {}
        
        # Get all period suffixes from first benchmark
        first_benchmark = list(all_benchmark_data.keys())[0]
        period_suffixes = list(all_benchmark_data[first_benchmark].keys())
        
        for period_suffix in period_suffixes:
            logger.info(f"Combining results for period: {period_suffix}")
            
            # Start with first benchmark's data
            first_benchmark = list(all_benchmark_data.keys())[0]
            combined_df = all_benchmark_data[first_benchmark][period_suffix].copy()
            
            # Join data from other benchmarks
            for benchmark in list(all_benchmark_data.keys())[1:]:
                if period_suffix in all_benchmark_data[benchmark]:
                    benchmark_df = all_benchmark_data[benchmark][period_suffix]
                    
                    # Select only benchmark-specific columns to avoid conflicts
                    # Exclude common columns like 'date' from joining
                    # Note: percentile columns excluded since they're handled by PER processor
                    common_cols = ['date']
                    benchmark_specific_cols = [col for col in benchmark_df.columns 
                                             if benchmark in col and col not in common_cols
                                             and not col.endswith('_percentile')]
                    
                    if benchmark_specific_cols:
                        combined_df = combined_df.join(
                            benchmark_df[benchmark_specific_cols], 
                            how='outer', 
                            rsuffix=f'_{benchmark}'
                        )
            
            combined_results[period_suffix] = combined_df
            logger.info(f"Combined {period_suffix}: {len(combined_df)} tickers, {len(combined_df.columns)} columns")
        
        return combined_results
    
    def _process_composite_rs_combined(self, composite_indices, timeframe, level, ticker_choice, benchmark_tickers):
        """
        Process RS analysis for composite indices with all benchmarks combined.
        
        Args:
            composite_indices: Dictionary of {group_name: price_series}
            timeframe: Data timeframe
            level: Analysis level ('sectors' or 'industries')
            ticker_choice: User ticker choice number
            benchmark_tickers: List of benchmark tickers
            
        Returns:
            List of created file paths
        """
        try:
            logger.info(f"Processing combined {level} RS analysis for {len(benchmark_tickers)} benchmarks...")
            
            # Initialize combined results container
            from .rs_base import RSResults
            combined_results = RSResults('ibd', level, timeframe)
            combined_results.metadata['benchmark_tickers'] = ';'.join(benchmark_tickers)
            combined_results.metadata['universe_size'] = len(composite_indices)
            
            # Collect results from all benchmarks
            all_benchmark_data = {}
            
            for benchmark_ticker in benchmark_tickers:
                logger.info(f"Processing {level} RS vs {benchmark_ticker}...")
                
                # Get analysis periods for this timeframe  
                periods = self.ibd_calculator.get_analysis_periods(timeframe)
                
                # Load benchmark data
                benchmark_data = self.ibd_calculator.load_benchmark_data(benchmark_ticker, timeframe)
                
                if benchmark_data is None:
                    logger.warning(f"No benchmark data for {level} RS analysis vs {benchmark_ticker}")
                    continue
                
                # Calculate RS for composite indices
                rs_data = self.ibd_calculator.calculate_composite_rs(
                    composite_indices, benchmark_data, periods, benchmark_ticker
                )
                
                if rs_data:
                    # Store RS data (percentiles will be calculated separately by PER processor)
                    all_benchmark_data[benchmark_ticker] = rs_data
                    logger.info(f"{level} RS vs {benchmark_ticker} calculated: {len(rs_data)} periods")
            
            # Combine all benchmark results
            if all_benchmark_data:
                combined_rs_data = self._combine_benchmark_results(all_benchmark_data)
                combined_results.rs_values = combined_rs_data
                
                # Save combined results (single file per asset class)
                # Use first benchmark ticker for naming combined files
                first_benchmark = benchmark_tickers[0] if benchmark_tickers else 'SPY'
                saved_files = self.ibd_calculator.save_rs_results(combined_results, level, str(ticker_choice), first_benchmark)
                
                logger.info(f"Combined {level} RS analysis completed: {len(saved_files)} files created for all benchmarks")
                return saved_files
            else:
                logger.warning(f"No {level} RS data calculated for any benchmark")
                return []
                
        except Exception as e:
            logger.error(f"Error processing combined {level} RS analysis: {e}")
            return []

    def _convert_composites_to_dataframe(self, composites_dict):
        """
        Convert composite indices from dict format to DataFrame format.
        Memory-optimized version.

        Args:
            composites_dict: Dictionary of {name: Series} composite indices

        Returns:
            DataFrame with dates as index and composite names as columns
        """
        if not composites_dict:
            return pd.DataFrame()

        # Convert dictionary of series to DataFrame efficiently
        try:
            result = pd.DataFrame(composites_dict)
            # Clear the input dict reference to help garbage collection
            composites_dict.clear()
            return result
        except Exception as e:
            logger.warning(f"Error converting composites to DataFrame: {e}")
            return pd.DataFrame()


def run_rs_analysis(ticker_list, config, user_config, ticker_choice=0, timeframes=None):
    """
    Standalone function to run complete RS analysis.

    Args:
        ticker_list: List of ticker symbols
        config: Config object
        user_config: User configuration object
        ticker_choice: User ticker choice number
        timeframes: List of timeframes to process (overrides individual RS flags if provided)

    Returns:
        Results summary dictionary
    """
    logger.info("Starting RS analysis pipeline...")
    
    # Check if any RS analysis is enabled
    rs_enabled = (getattr(user_config, 'rs_enable_stocks', True) or 
                  getattr(user_config, 'rs_enable_sectors', False) or 
                  getattr(user_config, 'rs_enable_industries', False))
    
    if not rs_enabled:
        logger.info("RS analysis disabled - skipping")
        return {'status': 'skipped', 'reason': 'RS analysis disabled in configuration'}
    
    # Create RS processor and run analysis
    rs_processor = RSProcessor(config, user_config)
    results = rs_processor.process_rs_analysis(ticker_list, ticker_choice, timeframes)

    # Generate summary
    summary = rs_processor.get_rs_summary(ticker_choice)
    results['summary'] = summary

    # Show final completion summary like streaming implementations
    logger.info(f"RS ANALYSIS COMPLETED!")
    logger.info(f"Total tickers processed: {results['universe_size']}")
    logger.info(f"Timeframes processed: {', '.join(results['timeframes_processed'])}")

    # Optional: Print summary for user feedback
    print(f"✅ RS analysis completed: {results['universe_size']} tickers, {len(results['timeframes_processed'])} timeframes")

    logger.info("RS analysis pipeline completed")
    return results