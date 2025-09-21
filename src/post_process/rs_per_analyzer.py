"""
RS/PER Multi-Timeframe Analysis Engine
=====================================

Performs comprehensive multi-timeframe analysis on RS and percentile data.
Calculates momentum metrics, patterns, market breadth, and classifications.

Key Analysis Components:
- Multi-timeframe momentum calculations (short/medium/long-term)
- Composite RS strength scoring
- Trend consistency analysis
- Market breadth and condition assessment
- Pattern classification and leadership analysis
"""

import pandas as pd
import numpy as np
from scipy import stats
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MomentumMetrics:
    """Container for momentum analysis results."""
    short_term: float  # 3d vs 7d
    medium_term: float  # 14d vs 44d
    long_term: float  # 66d vs 252d
    pattern: str
    classification: str


@dataclass
class MarketCondition:
    """Container for market breadth analysis."""
    total_stocks: int
    strong_rs_stocks: int
    consistent_stocks: int
    elite_stocks: int
    market_breadth_pct: float
    consistency_breadth_pct: float
    elite_breadth_pct: float
    condition: str
    description: str


class RSPERAnalyzer:
    """
    Main analysis engine for RS/PER multi-timeframe analysis.
    """

    def __init__(self, config=None):
        """Initialize the analyzer."""
        self.config = config

        # Define timeframe relationships for momentum calculations
        self.momentum_pairs = {
            'short': ('3d', '7d'),
            'medium': ('14d', '44d'),
            'long': ('66d', '252d')
        }

        # Classification thresholds
        self.rs_strength_threshold = 1.05
        self.trend_consistency_threshold = 75.0
        self.elite_percentile_threshold = 90.0

    def perform_multi_timeframe_analysis(self, data: Dict) -> Dict:
        """
        Perform comprehensive multi-timeframe analysis.

        Args:
            data: Loaded RS/PER data dictionary

        Returns:
            Complete analysis results dictionary
        """
        logger.info("Starting multi-timeframe analysis")

        results = {
            'date': data['date'],
            'analysis_timestamp': pd.Timestamp.now(),
            'stocks_analysis': None,
            'sectors_analysis': None,
            'industries_analysis': None,
            'market_condition': None,
            'summary_stats': {}
        }

        try:
            # Analyze stocks
            if 'stocks' in data['merged_data']:
                logger.info("Analyzing stocks data")
                stocks_analysis = self._analyze_stocks(data['merged_data']['stocks'])
                results['stocks_analysis'] = stocks_analysis

            # Analyze sectors
            if 'sectors' in data['merged_data']:
                logger.info("Analyzing sectors data")
                sectors_analysis = self._analyze_sectors(data['merged_data']['sectors'])
                results['sectors_analysis'] = sectors_analysis

            # Analyze industries
            if 'industries' in data['merged_data']:
                logger.info("Analyzing industries data")
                industries_analysis = self._analyze_industries(data['merged_data']['industries'])
                results['industries_analysis'] = industries_analysis

            # Calculate market condition
            if results['stocks_analysis']:
                logger.info("Calculating market condition")
                market_condition = self._calculate_market_condition(results['stocks_analysis'])
                results['market_condition'] = market_condition

            # Generate summary statistics
            results['summary_stats'] = self._generate_summary_stats(results)

            logger.info("Multi-timeframe analysis completed successfully")
            return results

        except Exception as e:
            logger.error(f"Multi-timeframe analysis failed: {e}")
            raise

    def _analyze_stocks(self, stocks_df: pd.DataFrame) -> Dict:
        """Analyze individual stock data."""
        analysis_results = []

        for _, stock in stocks_df.iterrows():
            try:
                # Extract RS values for all timeframes
                rs_values = self._extract_rs_values(stock)

                # Calculate momentum metrics
                momentum = self._calculate_momentum_metrics(rs_values)

                # Calculate composite metrics
                composite_strength = self._calculate_composite_strength(rs_values)
                trend_consistency = self._calculate_trend_consistency(rs_values)

                # Classification
                pattern = self._classify_momentum_pattern(momentum)
                leadership_type = self._determine_leadership_type(rs_values)

                # Extract percentile values
                percentile_values = self._extract_percentile_values(stock)

                stock_analysis = {
                    'ticker': stock.get('ticker', 'N/A'),
                    'rs_values': rs_values,
                    'percentile_values': percentile_values,
                    'momentum_short': momentum.short_term,
                    'momentum_medium': momentum.medium_term,
                    'momentum_long': momentum.long_term,
                    'momentum_pattern': momentum.pattern,
                    'composite_strength': composite_strength,
                    'trend_consistency': trend_consistency,
                    'leadership_type': leadership_type,
                    'classification': self._classify_stock_performance(
                        composite_strength, trend_consistency, percentile_values
                    )
                }

                analysis_results.append(stock_analysis)

            except Exception as e:
                logger.warning(f"Failed to analyze stock {stock.get('ticker', 'unknown')}: {e}")
                continue

        return {
            'individual_analysis': analysis_results,
            'summary': self._summarize_stock_analysis(analysis_results),
            'top_performers': self._identify_top_performers(analysis_results),
            'leader_laggard_analysis': self._analyze_leaders_laggards(analysis_results)
        }

    def _analyze_sectors(self, sectors_df: pd.DataFrame) -> Dict:
        """Analyze sector-level data."""
        analysis_results = []

        for _, sector in sectors_df.iterrows():
            try:
                # Extract RS values
                rs_values = self._extract_rs_values(sector)

                # Calculate momentum metrics
                momentum = self._calculate_momentum_metrics(rs_values)

                # Sector-specific analysis
                sector_analysis = {
                    'sector': sector.get('ticker', 'N/A'),
                    'rs_values': rs_values,
                    'momentum_short': momentum.short_term,
                    'momentum_medium': momentum.medium_term,
                    'momentum_long': momentum.long_term,
                    'momentum_pattern': momentum.pattern,
                    'composite_strength': self._calculate_composite_strength(rs_values),
                    'trend_consistency': self._calculate_trend_consistency(rs_values),
                    'rotation_signal': self._determine_rotation_signal(momentum),
                    'classification': self._classify_sector_performance(rs_values, momentum)
                }

                analysis_results.append(sector_analysis)

            except Exception as e:
                logger.warning(f"Failed to analyze sector {sector.get('ticker', 'unknown')}: {e}")
                continue

        return {
            'individual_analysis': analysis_results,
            'rotation_matrix': self._create_rotation_matrix(analysis_results),
            'quadrant_analysis': self._analyze_sector_quadrants(analysis_results)
        }

    def _analyze_industries(self, industries_df: pd.DataFrame) -> Dict:
        """Analyze industry-level data."""
        analysis_results = []

        for _, industry in industries_df.iterrows():
            try:
                # Extract RS values
                rs_values = self._extract_rs_values(industry)

                # Calculate momentum metrics
                momentum = self._calculate_momentum_metrics(rs_values)

                # Industry-specific analysis
                industry_analysis = {
                    'industry': industry.get('ticker', 'N/A'),
                    'industry_short_name': self._get_short_industry_name(industry.get('ticker', 'N/A')),
                    'rs_values': rs_values,
                    'momentum_short': momentum.short_term,
                    'momentum_medium': momentum.medium_term,
                    'momentum_long': momentum.long_term,
                    'momentum_pattern': momentum.pattern,
                    'composite_strength': self._calculate_composite_strength(rs_values),
                    'trend_consistency': self._calculate_trend_consistency(rs_values),
                    'rotation_signal': self._determine_rotation_signal(momentum),
                    'classification': self._classify_industry_performance(rs_values, momentum),
                    'market_cap_weight': 1.0,  # Default weight, could be enhanced
                    'stock_count': 1  # Default count, could be enhanced
                }

                analysis_results.append(industry_analysis)

            except Exception as e:
                logger.warning(f"Failed to analyze industry {industry.get('ticker', 'unknown')}: {e}")
                continue

        return {
            'individual_analysis': analysis_results,
            'rotation_matrix': self._create_rotation_matrix(analysis_results),
            'quadrant_analysis': self._analyze_industry_quadrants(analysis_results)
        }

    def _extract_rs_values(self, entity) -> Dict[str, float]:
        """Extract RS values for all timeframes."""
        rs_values = {}
        timeframes = ['3d', '5d', '7d', '14d', '22d', '44d', '66d', '132d', '252d']

        for tf in timeframes:
            rs_col = f"rs_{tf}"
            if rs_col in entity.index:
                value = entity[rs_col]
                rs_values[tf] = value if pd.notna(value) else 1.0
            else:
                rs_values[tf] = 1.0  # Default neutral value

        return rs_values

    def _extract_percentile_values(self, entity) -> Dict[str, float]:
        """Extract percentile values for all timeframes."""
        percentile_values = {}
        timeframes = ['3d', '5d', '7d', '14d', '22d', '44d', '66d', '132d', '252d']

        for tf in timeframes:
            per_col = f"daily_daily_*_{tf}_rs_vs_QQQ_per_NASDAQ100"
            # Find the actual column name (pattern matching)
            matching_cols = [col for col in entity.index if f"_{tf}_rs_vs_QQQ_per_NASDAQ100" in col]
            if matching_cols:
                value = entity[matching_cols[0]]
                percentile_values[tf] = value if pd.notna(value) else 50.0
            else:
                percentile_values[tf] = 50.0  # Default median value

        return percentile_values

    def _calculate_momentum_metrics(self, rs_values: Dict[str, float]) -> MomentumMetrics:
        """Calculate momentum metrics for different timeframes."""
        try:
            # Short-term momentum (3d vs 7d)
            momentum_short = ((rs_values['3d'] / rs_values['7d']) - 1) * 100 if rs_values['7d'] != 0 else 0

            # Medium-term momentum (14d vs 44d)
            momentum_medium = ((rs_values['14d'] / rs_values['44d']) - 1) * 100 if rs_values['44d'] != 0 else 0

            # Long-term momentum (66d vs 252d)
            momentum_long = ((rs_values['66d'] / rs_values['252d']) - 1) * 100 if rs_values['252d'] != 0 else 0

            # Classify pattern
            pattern = self._classify_momentum_pattern_from_values(momentum_short, momentum_medium, momentum_long)

            # Overall classification
            classification = self._classify_overall_momentum(momentum_short, momentum_medium, momentum_long)

            return MomentumMetrics(
                short_term=momentum_short,
                medium_term=momentum_medium,
                long_term=momentum_long,
                pattern=pattern,
                classification=classification
            )

        except Exception as e:
            logger.warning(f"Failed to calculate momentum metrics: {e}")
            return MomentumMetrics(0.0, 0.0, 0.0, "NEUTRAL", "NEUTRAL")

    def _calculate_composite_strength(self, rs_values: Dict[str, float]) -> float:
        """Calculate composite RS strength using geometric mean."""
        try:
            valid_values = [v for v in rs_values.values() if v > 0]
            if len(valid_values) == 0:
                return 1.0

            return stats.gmean(valid_values)

        except Exception as e:
            logger.warning(f"Failed to calculate composite strength: {e}")
            return 1.0

    def _calculate_trend_consistency(self, rs_values: Dict[str, float]) -> float:
        """Calculate trend consistency (% timeframes with RS > 1.0)."""
        try:
            strong_timeframes = sum(1 for v in rs_values.values() if v > 1.0)
            total_timeframes = len(rs_values)

            return (strong_timeframes / total_timeframes) * 100 if total_timeframes > 0 else 0.0

        except Exception as e:
            logger.warning(f"Failed to calculate trend consistency: {e}")
            return 0.0

    def _classify_momentum_pattern_from_values(self, short: float, medium: float, long: float) -> str:
        """Classify momentum pattern based on timeframe values."""
        if short > 0 and medium > 0 and long > 0:
            return "ACCELERATING"
        elif short > 0 and medium > 0:
            return "BUILDING"
        elif short < 0 and medium < 0 and long < 0:
            return "DECELERATING"
        elif short < 0 and medium < 0:
            return "WEAKENING"
        elif abs(short) < 2 and abs(medium) < 2 and abs(long) < 2:
            return "CONSOLIDATING"
        else:
            return "MIXED"

    def _classify_overall_momentum(self, short: float, medium: float, long: float) -> str:
        """Classify overall momentum trend."""
        avg_momentum = (short + medium + long) / 3

        if avg_momentum > 5:
            return "STRONG_BULLISH"
        elif avg_momentum > 0:
            return "BULLISH"
        elif avg_momentum > -5:
            return "NEUTRAL"
        else:
            return "BEARISH"

    def _determine_rotation_signal(self, momentum: MomentumMetrics) -> str:
        """Determine rotation signal for sectors/industries."""
        if momentum.short_term > 0 and momentum.medium_term > 0:
            return "STRONG_IN"
        elif momentum.short_term < 0 and momentum.medium_term < 0:
            return "ROTATING_OUT"
        else:
            return "NEUTRAL"

    def _classify_stock_performance(self, composite_strength: float, trend_consistency: float,
                                   percentile_values: Dict[str, float]) -> str:
        """Classify individual stock performance."""
        # Get average percentile
        avg_percentile = np.mean(list(percentile_values.values()))

        if composite_strength > 1.05 and trend_consistency > 75 and avg_percentile > 80:
            return "ELITE_LEADER"
        elif composite_strength > 1.05 and trend_consistency > 50:
            return "STRONG_PERFORMER"
        elif composite_strength > 1.0 and avg_percentile > 60:
            return "ABOVE_AVERAGE"
        elif composite_strength < 0.95 and trend_consistency < 25:
            return "WEAK_LAGGARD"
        else:
            return "AVERAGE"

    def _classify_sector_performance(self, rs_values: Dict[str, float], momentum: MomentumMetrics) -> str:
        """Classify sector performance."""
        composite_strength = self._calculate_composite_strength(rs_values)

        if composite_strength > 1.05 and momentum.medium_term > 0:
            return "LEADING"
        elif composite_strength > 1.0:
            return "OUTPERFORMING"
        elif composite_strength < 0.95:
            return "LAGGING"
        else:
            return "NEUTRAL"

    def _classify_industry_performance(self, rs_values: Dict[str, float], momentum: MomentumMetrics) -> str:
        """Classify industry performance."""
        return self._classify_sector_performance(rs_values, momentum)  # Same logic

    def _calculate_market_condition(self, stocks_analysis: Dict) -> MarketCondition:
        """Calculate overall market condition and breadth."""
        try:
            individual_analysis = stocks_analysis['individual_analysis']
            total_stocks = len(individual_analysis)

            if total_stocks == 0:
                return MarketCondition(0, 0, 0, 0, 0.0, 0.0, 0.0, "UNKNOWN", "No data available")

            # Count strong performers
            strong_rs_stocks = sum(1 for stock in individual_analysis
                                 if stock['composite_strength'] > self.rs_strength_threshold)

            # Count consistent performers
            consistent_stocks = sum(1 for stock in individual_analysis
                                  if stock['trend_consistency'] > self.trend_consistency_threshold)

            # Count elite performers (high percentile)
            elite_stocks = sum(1 for stock in individual_analysis
                             if np.mean(list(stock['percentile_values'].values())) >= self.elite_percentile_threshold)

            # Calculate percentages
            market_breadth_pct = (strong_rs_stocks / total_stocks) * 100
            consistency_breadth_pct = (consistent_stocks / total_stocks) * 100
            elite_breadth_pct = (elite_stocks / total_stocks) * 100

            # Determine market condition
            if market_breadth_pct > 50 and consistency_breadth_pct > 30:
                condition = "BROADLY_BULLISH"
                description = "Strong market with broad participation"
            elif market_breadth_pct > 30 and consistency_breadth_pct > 20:
                condition = "SELECTIVELY_BULLISH"
                description = "Moderate market strength with selective opportunities"
            elif market_breadth_pct > 20:
                condition = "MIXED_DEFENSIVE"
                description = "Mixed market conditions, defensive positioning recommended"
            else:
                condition = "BEARISH_RISK_OFF"
                description = "Weak market conditions, risk-off positioning"

            return MarketCondition(
                total_stocks=total_stocks,
                strong_rs_stocks=strong_rs_stocks,
                consistent_stocks=consistent_stocks,
                elite_stocks=elite_stocks,
                market_breadth_pct=market_breadth_pct,
                consistency_breadth_pct=consistency_breadth_pct,
                elite_breadth_pct=elite_breadth_pct,
                condition=condition,
                description=description
            )

        except Exception as e:
            logger.error(f"Failed to calculate market condition: {e}")
            return MarketCondition(0, 0, 0, 0, 0.0, 0.0, 0.0, "ERROR", f"Calculation failed: {e}")

    def _get_short_industry_name(self, industry_name: str) -> str:
        """Get shortened industry name for display."""
        # Simple shortening logic
        if len(industry_name) <= 15:
            return industry_name

        # Common abbreviations
        abbreviations = {
            'software': 'SW',
            'technology': 'Tech',
            'pharmaceuticals': 'Pharma',
            'biotechnology': 'Biotech',
            'semiconductors': 'Semis',
            'telecommunications': 'Telecom',
            'information': 'Info',
            'services': 'Svc',
            'manufacturing': 'Mfg'
        }

        short_name = industry_name.lower()
        for full, abbrev in abbreviations.items():
            short_name = short_name.replace(full, abbrev)

        return short_name[:15].title()

    def _analyze_sector_quadrants(self, analysis_results: List[Dict]) -> Dict:
        """Analyze sectors by RRG quadrants."""
        quadrants = {
            'leading_improving': [],
            'leading_weakening': [],
            'lagging_improving': [],
            'lagging_weakening': []
        }

        for sector in analysis_results:
            rs_strength = sector['composite_strength']
            momentum = sector['momentum_medium']

            if rs_strength > 1.0 and momentum > 0:
                quadrants['leading_improving'].append(sector)
            elif rs_strength > 1.0 and momentum < 0:
                quadrants['leading_weakening'].append(sector)
            elif rs_strength < 1.0 and momentum > 0:
                quadrants['lagging_improving'].append(sector)
            else:
                quadrants['lagging_weakening'].append(sector)

        return quadrants

    def _analyze_industry_quadrants(self, analysis_results: List[Dict]) -> Dict:
        """Analyze industries by RRG quadrants."""
        return self._analyze_sector_quadrants(analysis_results)  # Same logic

    def _create_rotation_matrix(self, analysis_results: List[Dict]) -> pd.DataFrame:
        """Create rotation matrix for sectors/industries."""
        matrix_data = []

        for entity in analysis_results:
            matrix_data.append({
                'entity': entity.get('sector', entity.get('industry', 'Unknown')),
                'composite_strength': entity['composite_strength'],
                'momentum_short': entity['momentum_short'],
                'momentum_medium': entity['momentum_medium'],
                'momentum_long': entity['momentum_long'],
                'rotation_signal': entity['rotation_signal'],
                'classification': entity['classification']
            })

        return pd.DataFrame(matrix_data)

    def _summarize_stock_analysis(self, analysis_results: List[Dict]) -> Dict:
        """Summarize stock analysis results."""
        if not analysis_results:
            return {}

        return {
            'total_stocks': len(analysis_results),
            'avg_composite_strength': np.mean([s['composite_strength'] for s in analysis_results]),
            'avg_trend_consistency': np.mean([s['trend_consistency'] for s in analysis_results]),
            'pattern_distribution': self._count_patterns(analysis_results, 'momentum_pattern'),
            'classification_distribution': self._count_patterns(analysis_results, 'classification')
        }

    def _count_patterns(self, analysis_results: List[Dict], field: str) -> Dict:
        """Count pattern distribution."""
        pattern_counts = {}
        for result in analysis_results:
            pattern = result.get(field, 'Unknown')
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        return pattern_counts

    def _identify_top_performers(self, analysis_results: List[Dict], top_n: int = 20) -> List[Dict]:
        """Identify top performing stocks."""
        sorted_stocks = sorted(analysis_results,
                             key=lambda x: (x['composite_strength'], x['trend_consistency']),
                             reverse=True)
        return sorted_stocks[:top_n]

    def _analyze_leaders_laggards(self, analysis_results: List[Dict]) -> Dict:
        """Analyze leaders vs laggards."""
        leaders = [s for s in analysis_results if s['composite_strength'] > 1.05]
        laggards = [s for s in analysis_results if s['composite_strength'] < 0.95]

        return {
            'leaders_count': len(leaders),
            'laggards_count': len(laggards),
            'leaders_pct': (len(leaders) / len(analysis_results)) * 100 if analysis_results else 0,
            'laggards_pct': (len(laggards) / len(analysis_results)) * 100 if analysis_results else 0,
            'top_leaders': sorted(leaders, key=lambda x: x['composite_strength'], reverse=True)[:10],
            'worst_laggards': sorted(laggards, key=lambda x: x['composite_strength'])[:10]
        }

    def _generate_summary_stats(self, results: Dict) -> Dict:
        """Generate overall summary statistics."""
        summary = {
            'analysis_date': results['date'],
            'analysis_timestamp': results['analysis_timestamp'],
            'data_quality': {}
        }

        # Add stock stats if available
        if results.get('stocks_analysis'):
            summary['stocks'] = results['stocks_analysis'].get('summary', {})

        # Add sector stats if available
        if results.get('sectors_analysis'):
            summary['sectors_count'] = len(results['sectors_analysis']['individual_analysis'])

        # Add industry stats if available
        if results.get('industries_analysis'):
            summary['industries_count'] = len(results['industries_analysis']['individual_analysis'])

        # Add market condition
        if results.get('market_condition'):
            summary['market_condition'] = {
                'condition': results['market_condition'].condition,
                'description': results['market_condition'].description,
                'breadth_pct': results['market_condition'].market_breadth_pct
            }

        return summary