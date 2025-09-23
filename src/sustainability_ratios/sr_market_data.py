"""
SR Market Data Loader Module
============================

Load market data for panels based on configuration.
Handles various data source types including tickers, ratios, and market indicators.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Union
from pathlib import Path

from ..indicators.indicator_parser import calculate_indicator

logger = logging.getLogger(__name__)


def decompose_data_source(data_source: str, panel_info: Dict) -> Dict[str, Union[str, List[str]]]:
    """
    Decompose data source into components for loading.

    Handles:
    - Simple tickers: QQQ → {type: 'ticker', ticker: 'QQQ'}
    - Bundled format: "QQQ + EMA(QQQ,10)" → {type: 'bundled', base_ticker: 'QQQ', components: [...]}
    - Enhanced format: PPO(12,26,9)_for_(QQQ) → {type: 'indicator', ticker: 'QQQ', indicator: 'PPO', params: {...}}
    - Ratios: XLY:XLP → {type: 'ratio', tickers: ['XLY', 'XLP']}

    Args:
        data_source: Original data source string
        panel_info: Panel information with parsed details

    Returns:
        Dict with decomposition information
    """
    try:
        # Check for bundled format
        if panel_info.get('is_bundled', False):
            bundled_components = panel_info.get('bundled_components', [])
            base_ticker = None
            indicators = []

            for component in bundled_components:
                if not ('(' in component and ')' in component):
                    # Simple ticker - this is our base
                    base_ticker = component.strip()
                else:
                    # Indicator component
                    indicators.append(component.strip())

            return {
                'type': 'bundled',
                'base_ticker': base_ticker,
                'indicators': indicators,
                'original_source': data_source
            }

        # Check for enhanced format indicators
        elif panel_info.get('has_indicator', False) and panel_info.get('tickers'):
            indicator = panel_info.get('indicator')
            tickers = panel_info.get('tickers', [])
            parameters = panel_info.get('parameters', {}) or panel_info.get('indicator_parameters', {})

            # For single-ticker indicators
            if len(tickers) == 1:
                return {
                    'type': 'indicator',
                    'ticker': tickers[0],
                    'indicator': indicator,
                    'parameters': parameters,
                    'original_source': data_source
                }
            # For multi-ticker indicators (like RATIO)
            else:
                return {
                    'type': 'multi_indicator',
                    'tickers': tickers,
                    'indicator': indicator,
                    'parameters': parameters,
                    'original_source': data_source
                }

        # Check for legacy ratio format
        elif ':' in data_source and not ('(' in data_source):
            parts = [p.strip() for p in data_source.split(':') if p.strip()]
            return {
                'type': 'ratio',
                'tickers': parts,
                'original_source': data_source
            }

        # Simple ticker
        else:
            return {
                'type': 'ticker',
                'ticker': data_source,
                'original_source': data_source
            }

    except Exception as e:
        logger.error(f"Error decomposing data source '{data_source}': {e}")
        # Fallback to simple ticker
        return {
            'type': 'ticker',
            'ticker': data_source,
            'original_source': data_source
        }


def load_market_data_for_panels(panel_config: Dict[str, Dict], data_reader) -> Dict[str, pd.DataFrame]:
    """
    Enhanced market data loader supporting all format types.

    Handles:
    - Simple tickers: QQQ, SPY
    - Enhanced indicators: PPO(12,26,9)_for_(QQQ), RSI(14)_for_(SPY)
    - Ticker-first indicators: PPO(QQQ,12,26,9), RSI(SPY,14)
    - Bundled format: "QQQ + EMA(QQQ,10)"
    - Ratios: XLY:XLP, RATIO_for_(SPY,QQQ)

    Args:
        panel_config: Panel configuration dict
        data_reader: DataReader instance

    Returns:
        Dict with computed market data for each data source
    """
    try:
        market_data = {}
        required_tickers = set()
        data_source_mappings = {}

        logger.info(f"Processing {len(panel_config)} panel configurations")

        # Phase 1: Decompose all data sources and collect required tickers
        for panel_name, panel_info in panel_config.items():
            data_source = panel_info.get('data_source')
            if not data_source:
                continue

            # Decompose data source into components
            decomposition = decompose_data_source(data_source, panel_info)
            data_source_mappings[data_source] = decomposition

            # Extract required tickers based on decomposition type
            if decomposition['type'] == 'ticker':
                required_tickers.add(decomposition['ticker'])
            elif decomposition['type'] == 'bundled':
                if decomposition['base_ticker']:
                    required_tickers.add(decomposition['base_ticker'])
            elif decomposition['type'] == 'indicator':
                required_tickers.add(decomposition['ticker'])
            elif decomposition['type'] in ['multi_indicator', 'ratio']:
                for ticker in decomposition['tickers']:
                    required_tickers.add(ticker)

        logger.info(f"Loading base data for {len(required_tickers)} tickers: {sorted(required_tickers)}")

        # Phase 2: Load base ticker data
        ticker_data = {}
        for ticker in required_tickers:
            try:
                data = data_reader.read_stock_data(ticker)
                if data is not None and not data.empty:
                    ticker_data[ticker] = data
                    logger.debug(f"Loaded data for {ticker}: {len(data)} rows")
                else:
                    logger.warning(f"No data available for {ticker}")

            except Exception as e:
                logger.warning(f"Error loading data for {ticker}: {e}")

        # Phase 3: Process each data source based on its type
        for data_source, decomposition in data_source_mappings.items():
            try:
                processed_data = process_decomposed_data_source(decomposition, ticker_data)
                if processed_data is not None:
                    market_data[data_source] = processed_data
                    logger.debug(f"Processed data source: {data_source}")
                else:
                    logger.warning(f"Failed to process data source: {data_source}")

            except Exception as e:
                logger.warning(f"Error processing data source {data_source}: {e}")

        logger.info(f"Successfully prepared market data for {len(market_data)}/{len(data_source_mappings)} data sources")
        return market_data

    except Exception as e:
        logger.error(f"Error loading market data for panels: {e}")
        return {}


def process_decomposed_data_source(decomposition: Dict, ticker_data: Dict[str, pd.DataFrame]) -> Optional[pd.DataFrame]:
    """
    Process decomposed data source and return computed data.

    Args:
        decomposition: Decomposed data source information
        ticker_data: Loaded ticker data

    Returns:
        Processed DataFrame or None if processing fails
    """
    try:
        data_type = decomposition['type']

        if data_type == 'ticker':
            # Simple ticker - return raw data
            ticker = decomposition['ticker']
            if ticker in ticker_data:
                return ticker_data[ticker].copy()
            else:
                logger.warning(f"No data available for ticker: {ticker}")
                return None

        elif data_type == 'indicator':
            # Single indicator calculation
            ticker = decomposition['ticker']
            indicator = decomposition['indicator']
            parameters = decomposition['parameters']

            if ticker not in ticker_data:
                logger.warning(f"No data available for ticker: {ticker}")
                return None

            # Get base data
            base_data = ticker_data[ticker]

            # Calculate indicator
            try:
                indicator_result = calculate_indicator_for_ticker(
                    base_data, indicator, parameters, ticker
                )
                return indicator_result
            except Exception as e:
                logger.error(f"Error calculating {indicator} for {ticker}: {e}")
                return None

        elif data_type == 'bundled':
            # Bundled format - combine base ticker with overlays
            base_ticker = decomposition['base_ticker']
            indicators = decomposition['indicators']

            if base_ticker not in ticker_data:
                logger.warning(f"No data available for base ticker: {base_ticker}")
                return None

            # Start with base data
            result_data = ticker_data[base_ticker].copy()

            # Add indicator overlays
            for indicator_str in indicators:
                try:
                    overlay_data = calculate_bundled_indicator(
                        indicator_str, ticker_data, base_ticker
                    )
                    if overlay_data is not None:
                        # print(f"📊 MERGING EMA DATA: {list(overlay_data.keys())}")  # Debug output
                        # Add overlay columns to result
                        for col_name, col_data in overlay_data.items():
                            result_data[col_name] = col_data
                            # print(f"   ✅ Added column: {col_name}")  # Debug output
                    else:
                        print(f"❌ NO OVERLAY DATA TO MERGE!")  # Keep - error reporting
                except Exception as e:
                    logger.warning(f"Error calculating bundled indicator {indicator_str}: {e}")

            # print(f"📈 FINAL RESULT_DATA COLUMNS: {list(result_data.columns)}")  # Debug output
            # print(f"📤 RETURNING RESULT_DATA TYPE: {type(result_data)}")  # Debug output
            return result_data

        elif data_type == 'ratio':
            # Legacy ratio calculation
            tickers = decomposition['tickers']
            if len(tickers) == 2:
                ratio_string = ':'.join(tickers)
                return calculate_ratio_data(ratio_string, ticker_data)
            else:
                logger.warning(f"Invalid ratio format - need exactly 2 tickers, got {len(tickers)}")
                return None

        elif data_type == 'multi_indicator':
            # Multi-ticker indicators (like RATIO_for_(SPY,QQQ))
            tickers = decomposition['tickers']
            indicator = decomposition['indicator']
            parameters = decomposition['parameters']

            # Load data for all tickers
            multi_data = {}
            for ticker in tickers:
                if ticker in ticker_data:
                    multi_data[ticker] = ticker_data[ticker]
                else:
                    logger.warning(f"Missing data for ticker in multi-indicator: {ticker}")

            if len(multi_data) != len(tickers):
                logger.warning(f"Incomplete data for multi-indicator {indicator}")
                return None

            # Calculate multi-ticker indicator
            try:
                return calculate_multi_ticker_indicator(
                    multi_data, indicator, parameters, tickers
                )
            except Exception as e:
                logger.error(f"Error calculating multi-ticker {indicator}: {e}")
                return None

        else:
            logger.error(f"Unknown data source type: {data_type}")
            return None

    except Exception as e:
        logger.error(f"Error processing decomposed data source: {e}")
        return None


def calculate_indicator_for_ticker(data: pd.DataFrame, indicator: str, parameters: Dict, ticker: str) -> Optional[pd.DataFrame]:
    """
    Calculate indicator for a single ticker.

    Args:
        data: Ticker OHLCV data
        indicator: Indicator name (PPO, RSI, EMA, etc.)
        parameters: Indicator parameters
        ticker: Ticker symbol for context

    Returns:
        DataFrame with indicator data or None if calculation fails
    """
    # DEBUG: Function entry logging
    logger.info(f"🚀 FUNCTION CALL: calculate_indicator_for_ticker()")
    logger.info(f"   ticker: '{ticker}'")
    logger.info(f"   indicator: '{indicator}'")
    logger.info(f"   parameters: {parameters}")
    logger.info(f"   data shape: {data.shape}")
    logger.info(f"   data columns: {list(data.columns)}")

    try:
        # Construct indicator string for calculation
        param_parts = []

        # Convert parameters dict to parameter string based on indicator type
        logger.info(f"🔧 PARAMETER CONSTRUCTION:")
        if indicator == 'PPO':
            fast = parameters.get('fast_period', 12)
            slow = parameters.get('slow_period', 26)
            signal = parameters.get('signal_period', 9)
            param_string = f"PPO({fast},{slow},{signal})"
            logger.info(f"   PPO parameters: fast={fast}, slow={slow}, signal={signal}")
        elif indicator == 'RSI':
            period = parameters.get('period', 14)
            param_string = f"RSI({period})"
            logger.info(f"   RSI parameters: period={period}")
        elif indicator == 'EMA':
            period = parameters.get('period', 20)
            param_string = f"EMA({period})"
            logger.info(f"   EMA parameters: period={period}")
        elif indicator == 'SMA':
            period = parameters.get('period', 20)
            param_string = f"SMA({period})"
            logger.info(f"   SMA parameters: period={period}")
        elif indicator == 'PRICE':
            logger.info(f"   PRICE indicator - returning original data")
            # PRICE indicator just returns the original data
            return data.copy()
        else:
            logger.error(f"❌ UNKNOWN INDICATOR TYPE: {indicator}")
            return None

        logger.info(f"   param_string: '{param_string}'")

        # Calculate using specialized functions to avoid parameter conflicts
        logger.info(f"📊 INDICATOR CALCULATION:")
        if indicator == 'EMA':
            logger.info(f"   Using specialized EMA calculation function")
            from ..indicators.MAs import calculate_ema_for_chart
            period = parameters.get('period', 20)
            logger.info(f"   Calling calculate_ema_for_chart(data, period={period})")
            result = calculate_ema_for_chart(data, period=period)
            logger.info(f"   EMA calculation result type: {type(result)}")
            logger.info(f"   EMA calculation result: {result}")
        elif indicator == 'SMA':
            logger.info(f"   Using specialized SMA calculation function")
            from ..indicators.MAs import calculate_sma_for_chart
            period = parameters.get('period', 20)
            logger.info(f"   Calling calculate_sma_for_chart(data, period={period})")
            result = calculate_sma_for_chart(data, period=period)
            logger.info(f"   SMA calculation result type: {type(result)}")
        else:
            logger.info(f"   Using generic indicator parser")
            logger.info(f"   Calling calculate_indicator(data, '{param_string}')")
            # Use generic indicator parser for other indicators
            result = calculate_indicator(data, param_string)
            logger.info(f"   Generic calculation result type: {type(result)}")

        if result is not None:
            logger.info(f"✅ CALCULATION SUCCESSFUL:")
            logger.info(f"   result type: {type(result)}")
            # Convert result to DataFrame if it's a dict
            if isinstance(result, dict):
                logger.info(f"   result is dict with keys: {list(result.keys())}")
                # Create DataFrame with original OHLCV data plus indicator columns
                result_df = data.copy()
                logger.info(f"   Starting with base data columns: {list(result_df.columns)}")

                for key, value in result.items():
                    logger.info(f"   Processing result key: '{key}' (type: {type(value)})")
                    if isinstance(value, (pd.Series, pd.DataFrame)):
                        if isinstance(value, pd.Series):
                            result_df[key] = value
                            logger.info(f"     Added Series column: '{key}' (length: {len(value)})")
                        else:
                            # Add all columns from DataFrame
                            for col in value.columns:
                                col_name = f"{key}_{col}"
                                result_df[col_name] = value[col]
                                logger.info(f"     Added DataFrame column: '{col}' → '{col_name}'")
                    else:
                        logger.info(f"     Skipped non-Series/DataFrame: '{key}' (type: {type(value)})")

                logger.info(f"   Final result_df columns: {list(result_df.columns)}")
                logger.info(f"   Final result_df shape: {result_df.shape}")
                return result_df
            else:
                logger.info(f"   result is not dict, returning directly")
                if hasattr(result, 'shape'):
                    logger.info(f"   result shape: {result.shape}")
                if hasattr(result, 'columns'):
                    logger.info(f"   result columns: {list(result.columns)}")
                return result
        else:
            logger.error(f"❌ CALCULATION FAILED: result is None")
            logger.error(f"   indicator: '{indicator}'")
            logger.error(f"   ticker: '{ticker}'")
            logger.error(f"   parameters: {parameters}")
            return None

    except Exception as e:
        import traceback
        logger.error(f"❌ EXCEPTION in calculate_indicator_for_ticker:")
        logger.error(f"   ticker: '{ticker}'")
        logger.error(f"   indicator: '{indicator}'")
        logger.error(f"   parameters: {parameters}")
        logger.error(f"   Exception type: {type(e).__name__}")
        logger.error(f"   Exception message: {str(e)}")
        logger.error(f"   Full traceback: {traceback.format_exc()}")
        return None


def calculate_bundled_indicator(indicator_str: str, ticker_data: Dict[str, pd.DataFrame], base_ticker: str) -> Optional[Dict[str, pd.Series]]:
    """
    Calculate indicator for bundled format.

    Args:
        indicator_str: Indicator string like "EMA(QQQ,10)"
        ticker_data: Available ticker data
        base_ticker: Base ticker for context

    Returns:
        Dict with indicator series or None if calculation fails
    """
    # DEBUG: Function entry logging
    # print(f"🚀 CALCULATE_BUNDLED_INDICATOR CALLED: {indicator_str}")  # Debug output
    logger.info(f"🚀 FUNCTION CALL: calculate_bundled_indicator()")
    logger.info(f"   indicator_str: '{indicator_str}'")
    logger.info(f"   base_ticker: '{base_ticker}'")
    logger.info(f"   ticker_data keys: {list(ticker_data.keys()) if ticker_data else 'None'}")

    try:
        from ..sustainability_ratios.enhanced_panel_parser import parse_enhanced_panel_entry

        # Parse the indicator string to extract components
        logger.info(f"📊 PARSING INDICATOR STRING:")
        parsed = parse_enhanced_panel_entry(indicator_str)
        logger.info(f"   parsed result: {parsed}")

        if not parsed or not parsed.get('has_indicator'):
            logger.error(f"❌ PARSING FAILED: Could not parse bundled indicator: {indicator_str}")
            logger.error(f"   parsed: {parsed}")
            logger.error(f"   has_indicator: {parsed.get('has_indicator') if parsed else 'None'}")
            return None

        indicator = parsed.get('indicator')
        parameters = parsed.get('parameters', {})
        tickers = parsed.get('tickers', [])

        logger.info(f"✅ PARSING SUCCESSFUL:")
        logger.info(f"   indicator: '{indicator}'")
        logger.info(f"   parameters: {parameters}")
        logger.info(f"   tickers: {tickers}")

        # Determine which ticker to use
        logger.info(f"🎯 TICKER DETERMINATION:")
        if tickers and len(tickers) > 0:
            ticker = tickers[0]
            logger.info(f"   Using ticker from parsed: '{ticker}'")
        else:
            ticker = base_ticker
            logger.info(f"   Using base_ticker: '{ticker}'")

        if ticker not in ticker_data:
            logger.error(f"❌ DATA MISSING: No data available for ticker {ticker} in bundled indicator")
            logger.error(f"   Available tickers: {list(ticker_data.keys())}")
            return None

        logger.info(f"✅ TICKER DATA FOUND: '{ticker}'")

        # Calculate the indicator
        data = ticker_data[ticker]
        logger.info(f"📈 CALCULATING INDICATOR:")
        logger.info(f"   data shape: {data.shape}")
        logger.info(f"   data columns: {list(data.columns)}")
        logger.info(f"   calling calculate_indicator_for_ticker with:")
        logger.info(f"     indicator: '{indicator}'")
        logger.info(f"     parameters: {parameters}")
        logger.info(f"     ticker: '{ticker}'")

        result = calculate_indicator_for_ticker(data, indicator, parameters, ticker)
        logger.info(f"   result type: {type(result)}")
        logger.info(f"   result is None: {result is None}")
        if result is not None:
            logger.info(f"   result shape: {result.shape}")
            logger.info(f"   result columns: {list(result.columns)}")

        if result is not None:
            logger.info(f"🎨 BUILDING OVERLAY DATA:")
            # Return as dict of series with appropriate column names
            overlay_data = {}
            for col in result.columns:
                if col not in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    # This is an indicator column
                    col_name = f"{indicator}_{col}" if col != indicator else indicator
                    overlay_data[col_name] = result[col]
                    logger.info(f"   Added overlay column: '{col}' → '{col_name}'")
                    logger.info(f"     series shape: {result[col].shape}")
                    logger.info(f"     series range: {result[col].min():.4f} to {result[col].max():.4f}")
                else:
                    logger.info(f"   Skipped OHLCV column: '{col}'")

            logger.info(f"✅ OVERLAY DATA COMPLETE:")
            logger.info(f"   overlay_data keys: {list(overlay_data.keys())}")
            logger.info(f"   total overlay columns: {len(overlay_data)}")

            # print(f"✅ BUNDLED_INDICATOR SUCCESS: {list(overlay_data.keys())}")  # Debug output
            return overlay_data
        else:
            print(f"❌ BUNDLED_INDICATOR FAILED: result is None")  # Keep - error reporting
            logger.error(f"❌ INDICATOR CALCULATION FAILED: result is None")
            logger.error(f"   indicator: '{indicator}'")
            logger.error(f"   parameters: {parameters}")
            logger.error(f"   ticker: '{ticker}'")
            return None

    except Exception as e:
        import traceback
        print(f"❌ BUNDLED_INDICATOR EXCEPTION: {e}")  # Keep - error reporting
        logger.error(f"❌ EXCEPTION in calculate_bundled_indicator:")
        logger.error(f"   indicator_str: '{indicator_str}'")
        logger.error(f"   base_ticker: '{base_ticker}'")
        logger.error(f"   Exception type: {type(e).__name__}")
        logger.error(f"   Exception message: {str(e)}")
        logger.error(f"   Full traceback: {traceback.format_exc()}")
        return None


def calculate_multi_ticker_indicator(data_dict: Dict[str, pd.DataFrame], indicator: str, parameters: Dict, tickers: List[str]) -> Optional[pd.DataFrame]:
    """
    Calculate indicators that require multiple tickers (like RATIO).

    Args:
        data_dict: Dict of ticker data
        indicator: Indicator name
        parameters: Indicator parameters
        tickers: List of ticker symbols

    Returns:
        Calculated indicator DataFrame or None if calculation fails
    """
    try:
        if indicator == 'RATIO' and len(tickers) == 2:
            # Calculate ratio between two tickers
            ticker1, ticker2 = tickers
            data1 = data_dict[ticker1]
            data2 = data_dict[ticker2]

            # Use existing ratio calculation
            ratio_string = f"{ticker1}:{ticker2}"
            return calculate_ratio_data(ratio_string, data_dict)
        else:
            logger.warning(f"Unsupported multi-ticker indicator: {indicator} with {len(tickers)} tickers")
            return None

    except Exception as e:
        logger.error(f"Error calculating multi-ticker indicator {indicator}: {e}")
        return None


def calculate_ratio_data(ratio_string: str, ticker_data: Dict[str, pd.DataFrame]) -> Optional[pd.DataFrame]:
    """
    Calculate ratio data from two tickers.

    Args:
        ratio_string: String like "XLY:XLP"
        ticker_data: Dict with ticker DataFrames

    Returns:
        DataFrame with ratio data (OHLCV format)
    """
    try:
        parts = ratio_string.split(':')
        if len(parts) != 2:
            logger.error(f"Invalid ratio format: {ratio_string}")
            return None

        numerator_ticker = parts[0].strip()
        denominator_ticker = parts[1].strip()

        if numerator_ticker not in ticker_data or denominator_ticker not in ticker_data:
            logger.warning(f"Missing data for ratio {ratio_string}")
            return None

        num_data = ticker_data[numerator_ticker]
        den_data = ticker_data[denominator_ticker]

        # Align data by index
        aligned_num, aligned_den = num_data.align(den_data, join='inner')

        if aligned_num.empty or aligned_den.empty:
            logger.warning(f"No overlapping data for ratio {ratio_string}")
            return None

        # Calculate ratio OHLCV
        ratio_data = pd.DataFrame(index=aligned_num.index)

        # Avoid division by zero
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in aligned_num.columns and col in aligned_den.columns:
                denominator = aligned_den[col].replace(0, np.nan)
                ratio_data[col] = aligned_num[col] / denominator

        # Volume is sum of both (approximate)
        if 'Volume' in aligned_num.columns and 'Volume' in aligned_den.columns:
            ratio_data['Volume'] = aligned_num['Volume'] + aligned_den['Volume']

        # Add metadata
        ratio_data.attrs = {
            'ratio_string': ratio_string,
            'numerator': numerator_ticker,
            'denominator': denominator_ticker,
            'data_type': 'ratio'
        }

        logger.debug(f"Calculated ratio data for {ratio_string}: {len(ratio_data)} rows")
        return ratio_data

    except Exception as e:
        logger.error(f"Error calculating ratio data for {ratio_string}: {e}")
        return None


def load_market_indicators(config) -> Dict[str, pd.Series]:
    """
    Load market indicators like VIX, Put/Call ratio, etc.

    Args:
        config: System configuration

    Returns:
        Dict with market indicator series
    """
    try:
        indicators = {}

        # List of market indicators to try to load
        indicator_tickers = ['VIX', 'CPCE', 'CORR', 'SKEW']

        for ticker in indicator_tickers:
            try:
                # Try to load from data reader (if available)
                if hasattr(config, 'data_reader') and config.data_reader:
                    data = config.data_reader.read_stock_data(ticker)
                    if data is not None and not data.empty and 'Close' in data.columns:
                        indicators[ticker] = data['Close']
                        logger.debug(f"Loaded market indicator {ticker}")

            except Exception as e:
                logger.debug(f"Could not load market indicator {ticker}: {e}")

        # Generate synthetic indicators if real data not available
        if not indicators:
            logger.info("Generating synthetic market indicators for testing")
            date_range = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')

            # Synthetic VIX (volatility)
            np.random.seed(42)
            vix_base = 20
            vix_noise = np.random.normal(0, 5, len(date_range))
            indicators['VIX'] = pd.Series(vix_base + vix_noise, index=date_range).clip(lower=10)

            # Synthetic Put/Call ratio
            pc_base = 0.8
            pc_noise = np.random.normal(0, 0.2, len(date_range))
            indicators['CPCE'] = pd.Series(pc_base + pc_noise, index=date_range).clip(lower=0.3, upper=2.0)

        logger.info(f"Loaded {len(indicators)} market indicators")
        return indicators

    except Exception as e:
        logger.error(f"Error loading market indicators: {e}")
        return {}


def validate_data_for_indicators(data: pd.DataFrame, indicator_requirements: Dict[str, int]) -> bool:
    """
    Validate that data meets requirements for indicator calculation.

    Args:
        data: Market data DataFrame
        indicator_requirements: Dict with indicator names and minimum data requirements

    Returns:
        True if data is valid
    """
    try:
        if data.empty:
            return False

        # Check basic OHLCV columns
        required_columns = ['Open', 'High', 'Low', 'Close']
        if not all(col in data.columns for col in required_columns):
            logger.warning("Missing required OHLCV columns")
            return False

        # Check data length against indicator requirements
        data_length = len(data)
        for indicator, min_length in indicator_requirements.items():
            if data_length < min_length:
                logger.warning(f"Insufficient data for {indicator}: need {min_length}, have {data_length}")
                return False

        # Check for excessive NaN values
        nan_pct = data['Close'].isna().sum() / len(data)
        if nan_pct > 0.1:  # More than 10% NaN
            logger.warning(f"Too many NaN values in data: {nan_pct:.1%}")
            return False

        return True

    except Exception as e:
        logger.error(f"Error validating data: {e}")
        return False


def get_data_summary(market_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, str]]:
    """
    Get summary information about loaded market data.

    Args:
        market_data: Dict with market data

    Returns:
        Dict with summary information
    """
    try:
        summary = {}

        for data_source, data in market_data.items():
            if data is not None and not data.empty:
                summary[data_source] = {
                    'rows': str(len(data)),
                    'start_date': str(data.index.min().date()) if not data.index.empty else 'N/A',
                    'end_date': str(data.index.max().date()) if not data.index.empty else 'N/A',
                    'columns': ', '.join(data.columns.tolist()),
                    'data_type': getattr(data, 'attrs', {}).get('data_type', 'ticker'),
                    'completeness': f"{(1 - data['Close'].isna().sum() / len(data)):.1%}" if 'Close' in data.columns else 'N/A'
                }
            else:
                summary[data_source] = {
                    'rows': '0',
                    'start_date': 'N/A',
                    'end_date': 'N/A',
                    'columns': 'N/A',
                    'data_type': 'empty',
                    'completeness': '0%'
                }

        return summary

    except Exception as e:
        logger.error(f"Error generating data summary: {e}")
        return {}