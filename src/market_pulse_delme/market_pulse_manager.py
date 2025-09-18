"""
Market Pulse Manager
===================

Pipeline-level orchestrator for comprehensive market timing and trend analysis.
Coordinates multiple indicator modules and provides market-wide insights.

Manages:
- GMI Calculator (General Market Index)
- FTD/DD Analyzer (Follow-Through/Distribution Days)
- Moving Average Analyzer (Chillax MA & Cycles)
- Breadth Analyzer (Net Highs/Lows, Universe filtering)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

from .calculators.gmi_calculator import GMICalculator
from .calculators.gmi2_calculator import GMI2Calculator
from .indicators.ftd_dd_analyzer import FTDDistributionAnalyzer
from .indicators.ma_cycles_analyzer import MACyclesAnalyzer
from .indicators.chillax_mas import ChillaxMAS
from .indicators.breadth_analyzer import BreadthAnalyzer

logger = logging.getLogger(__name__)


class MarketPulseManager:
    """
    Pipeline-level orchestrator for market timing and trend analysis.
    
    Coordinates multiple indicator modules and provides comprehensive
    market analysis separate from individual stock screening.
    """
    
    def __init__(self, config, user_config, data_reader=None):
        """
        Initialize Market Pulse Manager.
        
        Args:
            config: System configuration object
            user_config: User configuration object
            data_reader: DataReader instance (optional, not used in current implementation)
        """
        self.config = config
        self.user_config = user_config
        self.data_reader = data_reader
        
        # Track saved files for reporting
        self.saved_files = []
        
        # Get GMI-specific indexes from new configuration
        self.gmi_index1 = getattr(user_config, 'market_pulse_gmi_index1', 'SPY')  # For R3
        self.gmi_index2 = getattr(user_config, 'market_pulse_gmi_index2', 'QQQ')  # For R4, R5
        self.gmi_mf_index = getattr(user_config, 'market_pulse_gmi_mf_index', 'SPY')  # For R6
        
        # Create list of unique GMI target indexes
        self.gmi_targets = list(set([self.gmi_index1, self.gmi_index2, self.gmi_mf_index]))
        
        # FTD/DD analyzer targets
        self.ftd_targets = self._parse_target_indexes(
            getattr(user_config, 'ftd_dd_indexes', 'SPY;QQQ;IWM')
        )
        
        # MA Cycles analyzer targets
        self.ma_cycles_targets = self._parse_target_indexes(
            getattr(user_config, 'ma_cycles_indexes', 'SPY;QQQ;IWM')
        )
        
        # Chillax MAS analyzer targets
        self.chillax_targets = self._parse_target_indexes(
            getattr(user_config, 'chillax_mas_indexes', 'SPY;QQQ;IWM')
        )
        
        # Breadth analyzer targets (market-wide analysis)
        self.breadth_targets = self._parse_target_indexes(
            getattr(user_config, 'market_pulse_breadth_indexes', 'SPY;QQQ;IWM')
        )
        
        # Combined target list for overall management
        all_targets = set(self.gmi_targets)
        all_targets.update(self.ftd_targets)
        all_targets.update(self.ma_cycles_targets)
        all_targets.update(self.chillax_targets)
        all_targets.update(self.breadth_targets)
        self.target_indexes = list(all_targets)
        
        # For output file naming
        self.safe_user_choice = str(user_config.ticker_choice).replace('-', '_')
        self.original_user_choice = str(user_config.ticker_choice)  # Keep original format for GMI files
        
        # Initialize all indicator modules
        self._init_indicators()
        
    
    def _parse_target_indexes(self, target_config):
        """
        Parse target indexes from configuration, handling both string and list formats.
        
        Args:
            target_config: Configuration value (string or list)
            
        Returns:
            List of target index symbols
        """
        if isinstance(target_config, str):
            return [idx.strip() for idx in target_config.split(';')]
        elif isinstance(target_config, list):
            return target_config
        else:
            return ['SPY', 'QQQ', 'IWM', '^DJI']  # Default fallback
        
    def _init_indicators(self):
        """Initialize all market pulse indicator modules."""
        try:
            # Data paths for indicators
            self.data_paths = {
                'source_market_data': str(self.config.directories['DAILY_DATA_DIR']),
                'basic_calculations': str(self.config.directories['BASIC_CALCULATION_DIR']),
                'results': str(self.config.directories['RESULTS_DIR'])
            }
            
            # Initialize single GMI calculator for new R1-R6 model
            gmi_threshold = int(getattr(self.user_config, 'market_pulse_gmi_threshold', 3))
            gmi_confirmation_days = int(getattr(self.user_config, 'market_pulse_gmi_confirmation_days', 2))
            
            # New R1-R6 model uses single calculator that handles multiple indexes internally
            self.gmi_calculator = GMICalculator(
                paths=self.data_paths,
                threshold=gmi_threshold,
                confirmation_days=gmi_confirmation_days,
                user_config=self.user_config
            )
            
            # Initialize GMI2 calculator if enabled
            gmi2_enabled = getattr(self.user_config, 'market_pulse_gmi2_enable', False)
            if gmi2_enabled:
                gmi2_threshold = int(getattr(self.user_config, 'market_pulse_gmi2_threshold', 5))
                gmi2_confirmation_days = int(getattr(self.user_config, 'market_pulse_gmi2_confirmation_days', 2))
                
                self.gmi2_calculator = GMI2Calculator(
                    paths=self.data_paths,
                    threshold=gmi2_threshold,
                    confirmation_days=gmi2_confirmation_days,
                    user_config=self.user_config
                )
                logger.info(f"GMI2 Calculator initialized with threshold: {gmi2_threshold}")
            else:
                self.gmi2_calculator = None
                logger.info("GMI2 Calculator disabled")
            
            # Initialize FTD/DD analyzer if enabled
            ftd_dd_enabled = getattr(self.user_config, 'market_pulse_ftd_dd_enable', False)
            if ftd_dd_enabled:
                self.ftd_dd_analyzer = FTDDistributionAnalyzer(
                    paths=self.data_paths,
                    user_config=self.user_config
                )
                logger.info("FTD/DD Analyzer initialized")
            else:
                self.ftd_dd_analyzer = None
                logger.info("FTD/DD Analyzer disabled")
            
            # Initialize MA Cycles analyzer if enabled
            ma_cycles_enabled = getattr(self.user_config, 'market_pulse_ma_cycles_enable', False)
            if ma_cycles_enabled:
                self.ma_cycles_analyzer = MACyclesAnalyzer(
                    target_indexes=self.ma_cycles_targets,
                    config=self.config,
                    user_config=self.user_config
                )
                logger.info("MA Cycles Analyzer initialized")
            else:
                self.ma_cycles_analyzer = None
                logger.info("MA Cycles Analyzer disabled")
            
            # Initialize Chillax MAS analyzer if enabled
            chillax_enabled = getattr(self.user_config, 'market_pulse_chillax_ma_enable', False)
            if chillax_enabled:
                self.chillax_analyzer = ChillaxMAS(
                    target_indexes=self.chillax_targets,
                    config=self.config,
                    user_config=self.user_config
                )
                logger.info("Chillax MAS Analyzer initialized")
            else:
                self.chillax_analyzer = None
                logger.info("Chillax MAS Analyzer disabled")
            
            # Initialize Breadth analyzer if enabled
            breadth_enabled = getattr(self.user_config, 'market_breadth_enable', False)
            if breadth_enabled:
                self.breadth_analyzer = BreadthAnalyzer(
                    target_indexes=self.breadth_targets,
                    config=self.config,
                    user_config=self.user_config
                )
                logger.info("Market Breadth Analyzer initialized")
            else:
                self.breadth_analyzer = None
                logger.info("Market Breadth Analyzer disabled")
            
            logger.info(f"Market pulse indicators initialized for {len(self.target_indexes)} indexes")
            
        except Exception as e:
            logger.error(f"Error initializing market pulse indicators: {e}")
            raise
    
    def _is_gmi_enabled_for_timeframe(self, timeframe: str) -> bool:
        """
        Check if GMI calculation is enabled for the specified timeframe.

        GMI is daily-only by design. Only runs for daily timeframe.

        Args:
            timeframe: Data timeframe ('daily', 'weekly', 'monthly')

        Returns:
            bool: True if GMI is enabled and timeframe is daily, False otherwise
        """
        # GMI only runs on daily timeframe
        if timeframe.lower() != 'daily':
            logger.debug(f"GMI skipped for {timeframe}: daily-only by design")
            return False

        # Check if GMI is globally enabled
        is_enabled = getattr(self.user_config, 'market_pulse_gmi_enable', False)
        logger.debug(f"GMI enabled for daily: {is_enabled}")
        return is_enabled
    
    def _is_gmi2_enabled_for_timeframe(self, timeframe: str) -> bool:
        """
        Check if GMI2 calculation is enabled for the specified timeframe.

        GMI2 is daily-only by design. Only runs for daily timeframe.

        Args:
            timeframe: Data timeframe ('daily', 'weekly', 'monthly')

        Returns:
            bool: True if GMI2 is enabled and timeframe is daily, False otherwise
        """
        # GMI2 only runs on daily timeframe
        if timeframe.lower() != 'daily':
            logger.debug(f"GMI2 skipped for {timeframe}: daily-only by design")
            return False

        # Check if GMI2 is globally enabled
        is_enabled = getattr(self.user_config, 'market_pulse_gmi2_enable', False)
        logger.debug(f"GMI2 enabled for daily: {is_enabled}")
        return is_enabled
    
    def run_complete_analysis(self, timeframe: str = 'daily', data_date: str = None) -> Dict[str, Any]:
        """
        Run comprehensive market pulse analysis for a timeframe.
        
        Args:
            timeframe: Data timeframe ('daily', 'weekly', 'monthly')
            data_date: Date from dataframe for output naming (from main.py centralized management)
            
        Returns:
            Dictionary containing complete market pulse results
        """
        try:
            logger.info(f"Starting market pulse analysis for {timeframe} timeframe")
            
            # Reset saved files for this analysis
            self.saved_files = []
            
            results = {
                'timestamp': datetime.now().isoformat(),
                'success': True,
                'timeframe': timeframe,
                'data_date': data_date,
                'indexes': {},
                'market_breadth': {},
                'market_summary': {},
                'alerts': [],
                'analysis_metadata': {
                    'indicators_run': (
                        ['GMI'] + 
                        (['GMI2'] if self.gmi2_calculator else []) +
                        (['FTD_DD'] if self.ftd_dd_analyzer else []) +
                        (['MA_Cycles'] if self.ma_cycles_analyzer else []) +
                        (['Chillax_MAS'] if self.chillax_analyzer else []) +
                        (['Market_Breadth'] if self.breadth_analyzer else [])
                    ),
                    'target_indexes': self.target_indexes
                }
            }
            
            # Run analysis for each target index
            for index in self.target_indexes:
                logger.info(f"Analyzing {index}...")
                index_results = self._analyze_single_index(index, timeframe, data_date)
                results['indexes'][index] = index_results
            
            # Market-wide breadth analysis not implemented yet
            breadth_results = {}
            
            # Generate market summary
            results['market_summary'] = {'status': 'GMI analysis only'}
            
            # No alerts implemented yet
            results['alerts'] = []
            
            logger.info(f"Market pulse analysis completed successfully for {timeframe}")
            
            return results
            
        except Exception as e:
            logger.error(f"Market pulse analysis failed for {timeframe}: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'success': False,
                'error': str(e),
                'timeframe': timeframe,
                'data_date': data_date,
                'indexes': {},
                'market_breadth': {},
                'market_summary': {},
                'alerts': []
            }
    
    def _analyze_single_index(self, index: str, timeframe: str, data_date: str = None) -> Dict[str, Any]:
        """
        Run analysis for a single market index - MINIMAL IMPLEMENTATION.
        Currently only supports GMI analysis.
        
        Args:
            index: Market index symbol (SPY, QQQ, IWM, ^DJI)
            timeframe: Data timeframe
            data_date: Date for output naming
            
        Returns:
            Dictionary containing analysis results for the index
        """
        index_results = {
            'symbol': index,
            'analysis_success': False,
            'gmi': {},
            'gmi2': {},  # GMI2 analysis results
            'ftd_dd': {},  # FTD/DD analysis results
            'ma_cycles': {},  # MA Cycles analysis results
            'chillax': {},  # Chillax MAS analysis results
            'breadth': {},  # Market Breadth analysis results
            'errors': []
        }
        
        try:
            # 1. GMI Analysis (clean R1-R6 model) - with timeframe-specific enablement
            # Note: New model runs once and generates proper output files
            if index == self.gmi_targets[0]:  # Run GMI analysis only once for the first target
                # Check if GMI is enabled for this specific timeframe
                gmi_enabled_for_timeframe = self._is_gmi_enabled_for_timeframe(timeframe)
                
                if gmi_enabled_for_timeframe:
                    try:
                        # Call with proper parameters for filename and output
                        gmi_results = self.gmi_calculator.run_gmi_analysis(
                            timeframe=timeframe,
                            user_choice=self.original_user_choice,
                            date_str=data_date
                        )
                        index_results['gmi'] = gmi_results
                        index_results['analysis_success'] = True  # Success if GMI works
                        logger.debug(f"GMI R1-R6 analysis completed for {timeframe} timeframe")
                        # Track saved files
                        output_file = gmi_results.get('output_file')
                        if output_file:
                            self.saved_files.append(output_file)
                            logger.info(f"GMI output file: {output_file}")
                        else:
                            logger.warning("GMI analysis completed but no output file returned")
                    except Exception as e:
                        logger.error(f"GMI analysis failed: {e}")
                        index_results['errors'].append(f"GMI: {str(e)}")
                        index_results['gmi'] = {'error': str(e)}
                else:
                    logger.info(f"GMI analysis skipped - disabled for {timeframe} timeframe")
                    index_results['gmi'] = {'status': f'disabled_for_{timeframe}'}
            else:
                # For other indexes, just reference that GMI is handled globally
                index_results['gmi'] = {'status': 'handled_globally_in_r1_r6_model'}
                index_results['analysis_success'] = True
            
            # 2. GMI2 Analysis (multi-SMA model) - with timeframe-specific enablement
            if index == self.gmi_targets[0] and self.gmi2_calculator:  # Run GMI2 analysis only once for the first target
                # Check if GMI2 is enabled for this specific timeframe
                gmi2_enabled_for_timeframe = self._is_gmi2_enabled_for_timeframe(timeframe)
                
                if gmi2_enabled_for_timeframe:
                    try:
                        # Call with proper parameters for filename and output
                        gmi2_results = self.gmi2_calculator.run_gmi2_analysis(
                            timeframe=timeframe,
                            user_choice=self.original_user_choice,
                            date_str=data_date
                        )
                        index_results['gmi2'] = gmi2_results
                        logger.debug(f"GMI2 multi-SMA analysis completed for {timeframe} timeframe")
                        # Track saved files - handle both single and multiple files
                        output_files = gmi2_results.get('output_files', [])
                        if output_files:
                            self.saved_files.extend(output_files)
                            logger.info(f"GMI2 output files: {len(output_files)} files created")
                            for output_file in output_files:
                                logger.debug(f"GMI2 file: {output_file}")
                        else:
                            # Fallback for backward compatibility
                            output_file = gmi2_results.get('output_file')
                            if output_file:
                                self.saved_files.append(output_file)
                                logger.info(f"GMI2 output file: {output_file}")
                            else:
                                logger.warning("GMI2 analysis completed but no output files returned")
                    except Exception as e:
                        logger.error(f"GMI2 analysis failed: {e}")
                        index_results['errors'].append(f"GMI2: {str(e)}")
                        index_results['gmi2'] = {'error': str(e)}
                else:
                    logger.info(f"GMI2 analysis skipped - disabled for {timeframe} timeframe")
                    index_results['gmi2'] = {'status': f'disabled_for_{timeframe}'}
            elif self.gmi2_calculator:
                # For other indexes, just reference that GMI2 is handled globally
                index_results['gmi2'] = {'status': 'handled_globally_in_multi_sma_model'}
            else:
                # GMI2 calculator not enabled
                index_results['gmi2'] = {'status': 'disabled'}
            
            # 3. FTD/DD Analysis (if enabled and index is in ftd_targets)
            if self.ftd_dd_analyzer and index in self.ftd_targets:
                try:
                    # FTD/DD analyzer runs on all its configured indexes
                    if index == self.ftd_targets[0]:  # Run once for all indexes
                        ftd_dd_results = self.ftd_dd_analyzer.run_ftd_dd_analysis(timeframe)
                        if ftd_dd_results.get('success'):
                            # Extract results for this specific index
                            index_ftd_results = ftd_dd_results['results_by_index'].get(index, {})
                            index_results['ftd_dd'] = index_ftd_results
                            # Track saved files
                            output_files = ftd_dd_results.get('output_files', [])
                            if output_files:
                                self.saved_files.extend(output_files)
                                logger.info(f"FTD/DD files: {len(output_files)} files created")
                        else:
                            index_results['ftd_dd'] = {'error': ftd_dd_results.get('error', 'Unknown error')}
                    else:
                        # For other indexes, just reference that FTD/DD is handled globally
                        index_results['ftd_dd'] = {'status': 'handled_globally_in_ftd_dd_analyzer'}
                        
                except Exception as e:
                    logger.error(f"FTD/DD analysis failed: {e}")
                    index_results['errors'].append(f"FTD/DD: {str(e)}")
                    index_results['ftd_dd'] = {'error': str(e)}
            else:
                index_results['ftd_dd'] = {'status': 'disabled_or_not_target'}
            
            # 4. MA Cycles Analysis (if enabled and index is in ma_cycles_targets)
            if self.ma_cycles_analyzer and index in self.ma_cycles_targets:
                try:
                    # MA Cycles analyzer runs once for all its configured indexes
                    if index == self.ma_cycles_targets[0]:  # Run once for all indexes
                        ma_cycles_results = self.ma_cycles_analyzer.run_analysis(timeframe, data_date)
                        if ma_cycles_results.get('success'):
                            # Extract results for this specific index
                            index_ma_results = ma_cycles_results['analysis_results'].get(index, {})
                            index_results['ma_cycles'] = index_ma_results
                            # Track saved files
                            output_file = ma_cycles_results.get('output_file')
                            if output_file:
                                self.saved_files.append(output_file)
                                logger.info(f"MA Cycles file: {output_file}")
                            chart_files = ma_cycles_results.get('chart_files', {})
                            # Handle both list and dict formats for chart_files
                            if isinstance(chart_files, list):
                                for chart_file in chart_files:
                                    self.saved_files.append(chart_file)
                            elif isinstance(chart_files, dict):
                                for chart_file in chart_files.values():
                                    self.saved_files.append(chart_file)
                        else:
                            index_results['ma_cycles'] = {'error': ma_cycles_results.get('error', 'Unknown error')}
                    else:
                        # For other indexes, just reference that MA Cycles is handled globally
                        index_results['ma_cycles'] = {'status': 'handled_globally_in_ma_cycles_analyzer'}
                        
                except Exception as e:
                    logger.error(f"MA Cycles analysis failed: {e}")
                    index_results['errors'].append(f"MA_Cycles: {str(e)}")
                    index_results['ma_cycles'] = {'error': str(e)}
            else:
                index_results['ma_cycles'] = {'status': 'disabled_or_not_target'}
            
            # 5. Chillax MAS Analysis (if enabled and index is in chillax_targets)
            if self.chillax_analyzer and index in self.chillax_targets:
                try:
                    # Chillax analyzer runs once for all its configured indexes
                    if index == self.chillax_targets[0]:  # Run once for all indexes
                        chillax_results = self.chillax_analyzer.run_analysis(timeframe, data_date)
                        if chillax_results.get('success'):
                            # Extract results for this specific index
                            index_chillax_results = chillax_results['analysis_results'].get(index, {})
                            index_results['chillax'] = index_chillax_results
                            # Track saved files
                            output_file = chillax_results.get('output_file')
                            if output_file:
                                self.saved_files.append(output_file)
                                logger.info(f"Chillax file: {output_file}")
                            chart_files = chillax_results.get('chart_files', {})
                            # Handle both list and dict formats for chart_files
                            if isinstance(chart_files, list):
                                for chart_file in chart_files:
                                    self.saved_files.append(chart_file)
                            elif isinstance(chart_files, dict):
                                for chart_file in chart_files.values():
                                    self.saved_files.append(chart_file)
                        else:
                            index_results['chillax'] = {'error': chillax_results.get('error', 'Unknown error')}
                    else:
                        # For other indexes, just reference that Chillax is handled globally
                        index_results['chillax'] = {'status': 'handled_globally_in_chillax_analyzer'}
                        
                except Exception as e:
                    logger.error(f"Chillax MAS analysis failed: {e}")
                    index_results['errors'].append(f"Chillax_MAS: {str(e)}")
                    index_results['chillax'] = {'error': str(e)}
            else:
                index_results['chillax'] = {'status': 'disabled_or_not_target'}
            
            # 6. Market Breadth Analysis (if enabled and index is in breadth_targets)
            if self.breadth_analyzer and index in self.breadth_targets:
                try:
                    # Breadth analyzer runs once for all its configured indexes
                    if index == self.breadth_targets[0]:  # Run once for all indexes
                        breadth_results = self.breadth_analyzer.run_analysis(timeframe, data_date)
                        if breadth_results.get('success'):
                            # Extract results for this specific index (breadth is market-wide, so assign to primary index)
                            index_breadth_results = breadth_results.get('breadth_data', {})
                            index_results['breadth'] = index_breadth_results
                            # Track saved files
                            output_file = breadth_results.get('output_file')
                            if output_file:
                                self.saved_files.append(output_file)
                                logger.info(f"Market Breadth file: {output_file}")
                            chart_files = breadth_results.get('chart_files', {})
                            # Handle both list and dict formats for chart_files
                            if isinstance(chart_files, list):
                                for chart_file in chart_files:
                                    self.saved_files.append(chart_file)
                            elif isinstance(chart_files, dict):
                                for chart_file in chart_files.values():
                                    self.saved_files.append(chart_file)
                            else:
                                logger.warning(f"Unexpected chart_files type: {type(chart_files)}")
                        else:
                            index_results['breadth'] = {'error': breadth_results.get('error', 'Unknown error')}
                    else:
                        # For other indexes, just reference that Breadth is handled globally
                        index_results['breadth'] = {'status': 'handled_globally_in_breadth_analyzer'}
                        
                except Exception as e:
                    logger.error(f"Market Breadth analysis failed: {e}")
                    index_results['errors'].append(f"Market_Breadth: {str(e)}")
                    index_results['breadth'] = {'error': str(e)}
            else:
                index_results['breadth'] = {'status': 'disabled_or_not_target'}
            
            return index_results
            
        except Exception as e:
            logger.error(f"Error analyzing index {index}: {e}")
            return {
                'symbol': index,
                'analysis_success': False,
                'error': str(e),
                'gmi': {},
                'gmi2': {},
                'ftd_dd': {},
                'ma_cycles': {},
                'chillax': {},
                'breadth': {}
            }

    def save_results(self, output_path, timeframe: str, data_date: str = None) -> List[str]:
        """
        Return list of files that were saved during analysis.
        Files are automatically saved by GMI calculator during analysis.
        
        Args:
            output_path: Output directory path (unused - for compatibility)
            timeframe: Data timeframe (unused - for compatibility)
            data_date: Date for file naming (unused - for compatibility)
            
        Returns:
            List of saved file paths from the analysis
        """
        return self.saved_files.copy()


    def get_market_summary(self) -> str:
        """Get a simple market summary for now."""
        return "Market pulse analysis functionality restored."


# Convenience function for external use
def run_market_pulse_analysis(config, user_config, data_reader=None, timeframe='daily', data_date=None):
    """
    Convenience function to run market pulse analysis.
    
    Args:
        config: System configuration
        user_config: User configuration
        data_reader: DataReader instance (optional)
        timeframe: Data timeframe
        data_date: Date from dataframe
        
    Returns:
        Market pulse analysis results
    """
    manager = MarketPulseManager(config, user_config, data_reader)
    return manager.run_complete_analysis(timeframe, data_date)