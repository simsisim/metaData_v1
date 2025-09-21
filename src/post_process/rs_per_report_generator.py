"""
RS/PER Comprehensive Report Generator
====================================

Generates narrative text and analysis for comprehensive multi-timeframe
market analysis reports based on RS and percentile data.

Creates professional investment analysis including:
- Executive Summary with market condition assessment
- Market structure and breadth analysis
- Sector rotation analysis and positioning
- Industry rotation insights
- Trading strategies by timeframe
- Investment recommendations and risk management
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ReportSection:
    """Container for individual report sections."""
    title: str
    content: str
    data_tables: Optional[List[pd.DataFrame]] = None
    key_insights: Optional[List[str]] = None


class RSPERReportGenerator:
    """
    Generates comprehensive narrative analysis for RS/PER reports.
    """

    def __init__(self, config=None):
        """Initialize the report generator."""
        self.config = config

    def generate_comprehensive_report(self, analysis_results: Dict, charts: Dict[str, str],
                                    date_str: str = None) -> Dict[str, ReportSection]:
        """
        Generate complete narrative report sections.

        Args:
            analysis_results: Results from RSPERAnalyzer
            charts: Generated chart file paths
            date_str: Date string for the report

        Returns:
            Dictionary of report sections
        """
        if date_str is None:
            date_str = datetime.now().strftime('%Y%m%d')

        logger.info(f"Generating comprehensive RS/PER report for {date_str}")

        try:
            report_sections = {}

            # 1. Executive Summary
            report_sections['executive_summary'] = self._generate_executive_summary(
                analysis_results, date_str
            )

            # 2. Market Structure Assessment
            report_sections['market_structure'] = self._generate_market_structure_analysis(
                analysis_results
            )

            # 3. Multi-Timeframe RS Analysis
            report_sections['rs_analysis'] = self._generate_rs_analysis(
                analysis_results
            )

            # 4. Sector Rotation Analysis
            report_sections['sector_rotation'] = self._generate_sector_rotation_analysis(
                analysis_results
            )

            # 5. Industry Rotation Analysis
            report_sections['industry_rotation'] = self._generate_industry_rotation_analysis(
                analysis_results
            )

            # 6. Momentum Pattern Analysis
            report_sections['momentum_analysis'] = self._generate_momentum_analysis(
                analysis_results
            )

            # 7. Leadership Analysis
            report_sections['leadership_analysis'] = self._generate_leadership_analysis(
                analysis_results
            )

            # 8. Elite Performance Analysis
            report_sections['elite_analysis'] = self._generate_elite_analysis(
                analysis_results
            )

            # 9. Trading Strategies & Investment Recommendations
            report_sections['trading_strategies'] = self._generate_trading_strategies(
                analysis_results
            )

            logger.info("Comprehensive report generation completed")
            return report_sections

        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            raise

    def _generate_executive_summary(self, analysis_results: Dict, date_str: str) -> ReportSection:
        """Generate executive summary section."""
        try:
            market_condition = analysis_results.get('market_condition')
            stocks_analysis = analysis_results.get('stocks_analysis', {})
            summary_stats = analysis_results.get('summary_stats', {})

            if not market_condition:
                return ReportSection("Executive Summary", "Market condition data unavailable.")

            content = f"""
EXECUTIVE SUMMARY - {date_str}
{'=' * 50}

MARKET CONDITION: {market_condition.condition.replace('_', ' ').title()}

{market_condition.description}

The comprehensive analysis encompasses {market_condition.total_stocks:,} stocks across 9 timeframes
(3d, 5d, 7d, 14d, 22d, 44d, 66d, 132d, 252d), providing deep insights into market structure,
momentum patterns, and investment opportunities.

MARKET BREADTH INDICATORS:
• Strong RS Stocks (>1.05): {market_condition.strong_rs_stocks:,}/{market_condition.total_stocks:,} ({market_condition.market_breadth_pct:.1f}%)
• Consistent Performers (>75% timeframes bullish): {market_condition.consistent_stocks:,}/{market_condition.total_stocks:,} ({market_condition.consistency_breadth_pct:.1f}%)
• Elite Performers (90th+ percentile): {market_condition.elite_stocks:,}/{market_condition.total_stocks:,} ({market_condition.elite_breadth_pct:.1f}%)

INVESTMENT IMPLICATIONS:
{self._generate_investment_implications(market_condition)}

KEY OPPORTUNITIES IDENTIFIED:
{self._identify_key_opportunities(analysis_results)}
"""

            key_insights = [
                f"Market condition: {market_condition.condition.replace('_', ' ').title()}",
                f"Breadth: {market_condition.market_breadth_pct:.1f}% showing strong RS",
                f"Elite performers: {market_condition.elite_breadth_pct:.1f}% in 90th+ percentile",
                f"Analysis covers {market_condition.total_stocks:,} stocks across 9 timeframes"
            ]

            return ReportSection(
                title="Executive Summary",
                content=content,
                key_insights=key_insights
            )

        except Exception as e:
            logger.error(f"Failed to generate executive summary: {e}")
            return ReportSection("Executive Summary", "Summary generation failed.")

    def _generate_market_structure_analysis(self, analysis_results: Dict) -> ReportSection:
        """Generate market structure and breadth analysis."""
        try:
            market_condition = analysis_results.get('market_condition')
            stocks_analysis = analysis_results.get('stocks_analysis', {})

            if not market_condition:
                return ReportSection("Market Structure", "Market structure data unavailable.")

            # Analyze leadership patterns
            leader_laggard = stocks_analysis.get('leader_laggard_analysis', {})

            content = f"""
MARKET STRUCTURE ASSESSMENT
{'=' * 50}

BREADTH ANALYSIS:
The market is displaying {market_condition.condition.lower().replace('_', ' ')} characteristics with
{market_condition.market_breadth_pct:.1f}% of stocks showing strong relative strength (RS > 1.05).

Market Participation Analysis:
• Total Stocks Analyzed: {market_condition.total_stocks:,}
• Strong RS Leaders: {market_condition.strong_rs_stocks:,} ({market_condition.market_breadth_pct:.1f}%)
• Trend Consistent Stocks: {market_condition.consistent_stocks:,} ({market_condition.consistency_breadth_pct:.1f}%)
• Elite Performers: {market_condition.elite_stocks:,} ({market_condition.elite_breadth_pct:.1f}%)

LEADERSHIP DISTRIBUTION:
{self._analyze_leadership_distribution(leader_laggard)}

CROSS-TIMEFRAME STRENGTH PATTERNS:
{self._analyze_cross_timeframe_patterns(stocks_analysis)}

MARKET CONDITION INTERPRETATION:
{self._interpret_market_condition(market_condition)}
"""

            return ReportSection(
                title="Market Structure Assessment",
                content=content
            )

        except Exception as e:
            logger.error(f"Failed to generate market structure analysis: {e}")
            return ReportSection("Market Structure", "Market structure analysis failed.")

    def _generate_sector_rotation_analysis(self, analysis_results: Dict) -> ReportSection:
        """Generate sector rotation analysis."""
        try:
            sectors_analysis = analysis_results.get('sectors_analysis', {})

            if not sectors_analysis:
                return ReportSection("Sector Rotation", "Sector data unavailable.")

            quadrant_analysis = sectors_analysis.get('quadrant_analysis', {})
            rotation_matrix = sectors_analysis.get('rotation_matrix')

            content = f"""
SECTOR ROTATION ANALYSIS
{'=' * 50}

SECTOR QUADRANT DISTRIBUTION:
{self._analyze_sector_quadrants(quadrant_analysis)}

ROTATION SIGNALS:
{self._generate_sector_rotation_signals(sectors_analysis)}

SECTOR INVESTMENT THEMES:
{self._generate_sector_investment_themes(quadrant_analysis)}

TACTICAL POSITIONING RECOMMENDATIONS:
{self._generate_sector_positioning_recommendations(quadrant_analysis)}
"""

            return ReportSection(
                title="Sector Rotation Analysis",
                content=content,
                data_tables=[rotation_matrix] if rotation_matrix is not None else None
            )

        except Exception as e:
            logger.error(f"Failed to generate sector rotation analysis: {e}")
            return ReportSection("Sector Rotation", "Sector rotation analysis failed.")

    def _generate_industry_rotation_analysis(self, analysis_results: Dict) -> ReportSection:
        """Generate industry rotation analysis."""
        try:
            industries_analysis = analysis_results.get('industries_analysis', {})

            if not industries_analysis:
                return ReportSection("Industry Rotation", "Industry data unavailable.")

            quadrant_analysis = industries_analysis.get('quadrant_analysis', {})
            rotation_matrix = industries_analysis.get('rotation_matrix')

            content = f"""
INDUSTRY ROTATION ANALYSIS
{'=' * 50}

INDUSTRY QUADRANT DISTRIBUTION:
{self._analyze_industry_quadrants(quadrant_analysis)}

GRANULAR ROTATION INSIGHTS:
{self._generate_industry_rotation_insights(industries_analysis)}

INDUSTRY INVESTMENT OPPORTUNITIES:
{self._identify_industry_opportunities(quadrant_analysis)}

SECTOR DRILL-DOWN ANALYSIS:
{self._generate_sector_drilldown(industries_analysis)}

RISK CONSIDERATIONS:
{self._generate_industry_risk_analysis(quadrant_analysis)}
"""

            return ReportSection(
                title="Industry Rotation Analysis",
                content=content,
                data_tables=[rotation_matrix] if rotation_matrix is not None else None
            )

        except Exception as e:
            logger.error(f"Failed to generate industry rotation analysis: {e}")
            return ReportSection("Industry Rotation", "Industry rotation analysis failed.")

    def _generate_momentum_analysis(self, analysis_results: Dict) -> ReportSection:
        """Generate momentum pattern analysis."""
        try:
            stocks_analysis = analysis_results.get('stocks_analysis', {})
            summary = stocks_analysis.get('summary', {})

            content = f"""
MOMENTUM PATTERN ANALYSIS
{'=' * 50}

PATTERN DISTRIBUTION:
{self._analyze_momentum_patterns(summary)}

TIMEFRAME MOMENTUM INSIGHTS:
{self._generate_timeframe_momentum_insights(stocks_analysis)}

MOMENTUM CLASSIFICATION ANALYSIS:
{self._analyze_momentum_classifications(summary)}

INVESTMENT TIMING IMPLICATIONS:
{self._generate_momentum_timing_insights(stocks_analysis)}
"""

            return ReportSection(
                title="Momentum Pattern Analysis",
                content=content
            )

        except Exception as e:
            logger.error(f"Failed to generate momentum analysis: {e}")
            return ReportSection("Momentum Analysis", "Momentum analysis failed.")

    def _generate_leadership_analysis(self, analysis_results: Dict) -> ReportSection:
        """Generate leadership analysis."""
        try:
            stocks_analysis = analysis_results.get('stocks_analysis', {})
            top_performers = stocks_analysis.get('top_performers', [])
            leader_laggard = stocks_analysis.get('leader_laggard_analysis', {})

            content = f"""
LEADERSHIP ANALYSIS
{'=' * 50}

TOP PERFORMING LEADERS:
{self._format_top_performers(top_performers[:10])}

LEADERSHIP CONSISTENCY METRICS:
{self._analyze_leadership_consistency(leader_laggard)}

MULTI-TIMEFRAME LEADERSHIP PATTERNS:
{self._analyze_multi_timeframe_leadership(stocks_analysis)}

LEADERSHIP SUSTAINABILITY ASSESSMENT:
{self._assess_leadership_sustainability(top_performers)}
"""

            return ReportSection(
                title="Leadership Analysis",
                content=content
            )

        except Exception as e:
            logger.error(f"Failed to generate leadership analysis: {e}")
            return ReportSection("Leadership Analysis", "Leadership analysis failed.")

    def _generate_elite_analysis(self, analysis_results: Dict) -> ReportSection:
        """Generate elite performance analysis."""
        try:
            stocks_analysis = analysis_results.get('stocks_analysis', {})
            individual_analysis = stocks_analysis.get('individual_analysis', [])

            # Filter elite performers
            elite_performers = [stock for stock in individual_analysis
                              if stock['classification'] in ['ELITE_LEADER', 'STRONG_PERFORMER']]

            content = f"""
ELITE PERFORMANCE ANALYSIS
{'=' * 50}

ELITE PERFORMERS IDENTIFIED: {len(elite_performers)}

{self._format_elite_performers(elite_performers[:6])}

ELITE PERFORMANCE CHARACTERISTICS:
{self._analyze_elite_characteristics(elite_performers)}

PERCENTILE RANKING ANALYSIS:
{self._analyze_percentile_rankings(elite_performers)}

PORTFOLIO IMPLICATIONS:
{self._generate_elite_portfolio_implications(elite_performers)}
"""

            return ReportSection(
                title="Elite Performance Analysis",
                content=content
            )

        except Exception as e:
            logger.error(f"Failed to generate elite analysis: {e}")
            return ReportSection("Elite Analysis", "Elite analysis failed.")

    def _generate_trading_strategies(self, analysis_results: Dict) -> ReportSection:
        """Generate trading strategies and investment recommendations."""
        try:
            content = f"""
TRADING STRATEGIES & INVESTMENT RECOMMENDATIONS
{'=' * 50}

TIMEFRAME-SPECIFIC STRATEGIES:

SHORT-TERM TRADING (3-7 Day Focus):
{self._generate_short_term_strategies(analysis_results)}

MEDIUM-TERM SWING TRADING (14-44 Day Focus):
{self._generate_medium_term_strategies(analysis_results)}

LONG-TERM INVESTMENT THEMES (66-252 Day Focus):
{self._generate_long_term_strategies(analysis_results)}

RISK MANAGEMENT GUIDELINES:
{self._generate_risk_management_guidelines(analysis_results)}

POSITION SIZING RECOMMENDATIONS:
{self._generate_position_sizing_recommendations(analysis_results)}

PORTFOLIO CONSTRUCTION INSIGHTS:
{self._generate_portfolio_construction_insights(analysis_results)}
"""

            return ReportSection(
                title="Trading Strategies & Investment Recommendations",
                content=content
            )

        except Exception as e:
            logger.error(f"Failed to generate trading strategies: {e}")
            return ReportSection("Trading Strategies", "Strategy generation failed.")

    # Helper methods for content generation

    def _generate_investment_implications(self, market_condition) -> str:
        """Generate investment implications based on market condition."""
        if market_condition.market_breadth_pct > 50:
            return """• Broad-based opportunity environment supports diversified long positions
• Focus on momentum continuation strategies
• Sector rotation strategies likely to be successful
• Higher risk tolerance appropriate given broad participation"""
        elif market_condition.market_breadth_pct > 30:
            return """• Selective opportunity environment requires careful stock selection
• Focus on high-conviction positions in strong performers
• Sector rotation mixed, concentrate on clear leaders
• Moderate risk tolerance with defensive hedges"""
        else:
            return """• Defensive positioning recommended
• Focus on preservation of capital and risk management
• Limited opportunities, concentrate only on highest-conviction names
• Lower risk tolerance, consider increased cash positions"""

    def _identify_key_opportunities(self, analysis_results: Dict) -> str:
        """Identify key investment opportunities."""
        opportunities = []

        # Check for strong sectors
        if 'sectors_analysis' in analysis_results:
            sectors = analysis_results['sectors_analysis'].get('individual_analysis', [])
            strong_sectors = [s for s in sectors if s['composite_strength'] > 1.05]
            if strong_sectors:
                top_sector = max(strong_sectors, key=lambda x: x['composite_strength'])
                opportunities.append(f"• Strong sector rotation into {top_sector['sector']}")

        # Check for momentum acceleration
        if 'stocks_analysis' in analysis_results:
            stocks = analysis_results['stocks_analysis'].get('individual_analysis', [])
            accelerating = [s for s in stocks if s['momentum_pattern'] == 'ACCELERATING']
            if accelerating:
                opportunities.append(f"• {len(accelerating)} stocks showing momentum acceleration")

        # Check for elite performers
        elite_count = len([s for s in stocks if s['classification'] == 'ELITE_LEADER'])
        if elite_count > 0:
            opportunities.append(f"• {elite_count} elite leaders identified for core positions")

        return '\n'.join(opportunities) if opportunities else "• Limited opportunities in current market environment"

    def _analyze_sector_quadrants(self, quadrant_analysis: Dict) -> str:
        """Analyze sector quadrant distribution."""
        if not quadrant_analysis:
            return "Quadrant analysis unavailable."

        leading_improving = len(quadrant_analysis.get('leading_improving', []))
        leading_weakening = len(quadrant_analysis.get('leading_weakening', []))
        lagging_improving = len(quadrant_analysis.get('lagging_improving', []))
        lagging_weakening = len(quadrant_analysis.get('lagging_weakening', []))

        return f"""• Leading & Improving: {leading_improving} sectors (buy/overweight candidates)
• Leading & Weakening: {leading_weakening} sectors (profit-taking candidates)
• Lagging & Improving: {lagging_improving} sectors (emerging opportunities)
• Lagging & Weakening: {lagging_weakening} sectors (avoid/underweight)"""

    def _analyze_industry_quadrants(self, quadrant_analysis: Dict) -> str:
        """Analyze industry quadrant distribution."""
        return self._analyze_sector_quadrants(quadrant_analysis)  # Same format

    def _format_top_performers(self, top_performers: List[Dict]) -> str:
        """Format top performers list."""
        if not top_performers:
            return "No top performers data available."

        formatted = []
        for i, stock in enumerate(top_performers, 1):
            formatted.append(
                f"{i:2d}. {stock['ticker']:8s} - Strength: {stock['composite_strength']:.3f}, "
                f"Consistency: {stock['trend_consistency']:.1f}%, "
                f"Pattern: {stock['momentum_pattern']}"
            )

        return '\n'.join(formatted)

    def _format_elite_performers(self, elite_performers: List[Dict]) -> str:
        """Format elite performers list."""
        if not elite_performers:
            return "No elite performers identified."

        formatted = []
        for i, stock in enumerate(elite_performers, 1):
            avg_percentile = np.mean(list(stock['percentile_values'].values()))
            formatted.append(
                f"{i}. {stock['ticker']:8s} - Classification: {stock['classification']}, "
                f"Avg Percentile: {avg_percentile:.1f}, "
                f"Strength: {stock['composite_strength']:.3f}"
            )

        return '\n'.join(formatted)

    def _generate_short_term_strategies(self, analysis_results: Dict) -> str:
        """Generate short-term trading strategies."""
        return """• Focus on momentum continuation plays with 3-7 day holding periods
• Target stocks showing ACCELERATING or BUILDING momentum patterns
• Use tight stops (3-5%) given short-term volatility
• Monitor for news catalysts and earnings reactions
• Best suited for experienced momentum traders"""

    def _generate_medium_term_strategies(self, analysis_results: Dict) -> str:
        """Generate medium-term trading strategies."""
        return """• Swing trading opportunities in sector rotation plays
• Target 14-44 day holding periods with sector ETF pairs trades
• Focus on stocks transitioning from lagging to leading quadrants
• Use moderate stops (7-10%) allowing for normal volatility
• Ideal for sector rotation and earnings cycle plays"""

    def _generate_long_term_strategies(self, analysis_results: Dict) -> str:
        """Generate long-term investment strategies."""
        return """• Build core positions in consistent elite leaders
• Focus on 66-252 day trend strength and sustainability
• Emphasize fundamental strength alongside technical momentum
• Use wider stops (12-15%) for long-term trend participation
• Suitable for growth-oriented portfolio construction"""

    def _generate_risk_management_guidelines(self, analysis_results: Dict) -> str:
        """Generate risk management guidelines."""
        market_condition = analysis_results.get('market_condition')
        if not market_condition:
            return "Risk guidelines unavailable."

        if market_condition.market_breadth_pct > 50:
            return """• Maximum position size: 5-8% per stock, 15-20% per sector
• Portfolio beta target: 1.0-1.3
• Stop-loss levels: 8-12% for individual positions
• Sector diversification: Maximum 25% in any single sector"""
        elif market_condition.market_breadth_pct > 30:
            return """• Maximum position size: 3-5% per stock, 12-15% per sector
• Portfolio beta target: 0.8-1.1
• Stop-loss levels: 6-10% for individual positions
• Sector diversification: Maximum 20% in any single sector"""
        else:
            return """• Maximum position size: 2-3% per stock, 8-10% per sector
• Portfolio beta target: 0.5-0.8
• Stop-loss levels: 5-8% for individual positions
• Sector diversification: Maximum 15% in any single sector"""

    def _generate_position_sizing_recommendations(self, analysis_results: Dict) -> str:
        """Generate position sizing recommendations."""
        return """• Elite Leaders: Full position size (maximum allocation)
• Strong Performers: 75% of maximum position size
• Above Average: 50% of maximum position size
• Average/Neutral: 25% of maximum position size or avoid
• Weak Laggards: Avoid or short consideration"""

    def _generate_portfolio_construction_insights(self, analysis_results: Dict) -> str:
        """Generate portfolio construction insights."""
        return """• Core Holdings (40-60%): Elite leaders with multi-timeframe strength
• Satellite Positions (20-30%): Sector rotation and momentum plays
• Opportunistic Trades (10-15%): Short-term momentum acceleration
• Cash/Hedges (10-20%): Risk management and opportunity reserves
• Regular rebalancing based on changing RS rankings and momentum patterns"""

    # Additional helper methods would continue here...
    def _analyze_leadership_distribution(self, leader_laggard: Dict) -> str:
        """Analyze leadership distribution patterns."""
        if not leader_laggard:
            return "Leadership distribution data unavailable."

        leaders_pct = leader_laggard.get('leaders_pct', 0)
        laggards_pct = leader_laggard.get('laggards_pct', 0)

        return f"""• Market Leaders (RS >1.05): {leaders_pct:.1f}% of universe
• Market Laggards (RS <0.95): {laggards_pct:.1f}% of universe
• Leadership Ratio: {leaders_pct/max(laggards_pct, 1):.2f}:1 (leaders to laggards)"""

    def _analyze_cross_timeframe_patterns(self, stocks_analysis: Dict) -> str:
        """Analyze cross-timeframe strength patterns."""
        return """• Multi-timeframe analysis reveals momentum consistency patterns
• Strong correlation between short-term and medium-term trends
• Long-term strength provides foundation for sustained moves
• Cross-timeframe divergences signal potential trend changes"""

    def _interpret_market_condition(self, market_condition) -> str:
        """Interpret overall market condition."""
        condition = market_condition.condition

        interpretations = {
            'BROADLY_BULLISH': "Favorable environment for growth strategies and momentum plays",
            'SELECTIVELY_BULLISH': "Stock picker's market with selective opportunities",
            'MIXED_DEFENSIVE': "Challenging environment requiring defensive positioning",
            'BEARISH_RISK_OFF': "Risk-off environment, focus on capital preservation"
        }

        return interpretations.get(condition, "Market condition requires careful analysis")

    def _generate_sector_rotation_signals(self, sectors_analysis: Dict) -> str:
        """Generate sector rotation signals."""
        individual_analysis = sectors_analysis.get('individual_analysis', [])

        strong_in = [s for s in individual_analysis if s.get('rotation_signal') == 'STRONG_IN']
        rotating_out = [s for s in individual_analysis if s.get('rotation_signal') == 'ROTATING_OUT']

        signals = []
        if strong_in:
            signals.append(f"• STRONG IN: {', '.join([s['sector'] for s in strong_in[:3]])}")
        if rotating_out:
            signals.append(f"• ROTATING OUT: {', '.join([s['sector'] for s in rotating_out[:3]])}")

        return '\n'.join(signals) if signals else "• Mixed rotation signals across sectors"

    def _generate_sector_investment_themes(self, quadrant_analysis: Dict) -> str:
        """Generate sector investment themes."""
        themes = []

        leading_improving = quadrant_analysis.get('leading_improving', [])
        if leading_improving:
            themes.append(f"• Growth Theme: {', '.join([s['sector'] for s in leading_improving[:2]])}")

        lagging_improving = quadrant_analysis.get('lagging_improving', [])
        if lagging_improving:
            themes.append(f"• Recovery Theme: {', '.join([s['sector'] for s in lagging_improving[:2]])}")

        return '\n'.join(themes) if themes else "• No clear thematic opportunities identified"

    def _generate_sector_positioning_recommendations(self, quadrant_analysis: Dict) -> str:
        """Generate sector positioning recommendations."""
        recommendations = []

        leading_improving = len(quadrant_analysis.get('leading_improving', []))
        leading_weakening = len(quadrant_analysis.get('leading_weakening', []))

        if leading_improving > 0:
            recommendations.append("• Overweight leading & improving sectors for momentum continuation")
        if leading_weakening > 0:
            recommendations.append("• Consider profit-taking in leading & weakening sectors")

        return '\n'.join(recommendations) if recommendations else "• Maintain neutral sector allocation"