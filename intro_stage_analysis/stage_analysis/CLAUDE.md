# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains a Stage Analysis trading indicator system based on TradingView Pine Script or similar financial analysis framework. The project implements a systematic approach to market stage classification using moving averages and volatility metrics.

## Core Concepts

### Stage Analysis Framework
The system implements an 8-stage market cycle analysis using three key moving averages:
- **EMA10**: 10-day Exponential Moving Average (short-term momentum)
- **SMA20**: 20-day Simple Moving Average (intermediate support/resistance)  
- **SMA50**: 50-day Simple Moving Average (long-term trend filter)

### ATR-to-SMA50 Ratio
A critical volatility measure that gauges the rate of price increase relative to the trend:
- **≤4**: Moderate/steady movement (Stages 2A/2B, 4A/4B)
- **>7**: Extended movement indicating potential exhaustion (Stage 2C)
- **<4**: Reduced volatility, possible trend stabilization (Stage 4C)

### Market Stages
The framework defines 8 distinct market stages across 4 main phases:

**Stage 1 - Basing Phase:**
- 1A (Upward Pivot): Price ≥ EMA10, Price ≤ SMA20, Price ≤ SMA50
- 1B (Mean Reversion): Price ≥ EMA10, Price ≥ SMA20, Price ≤ SMA50

**Stage 2 - Advancing Phase:**
- 2A (Bullish Trend): Price above all MAs, ATR-to-SMA50 ≤ 4
- 2B (Breakout Confirmation): Price above all MAs with proper MA stacking, ATR-to-SMA50 ≤ 4
- 2C (Bullish Extended): Price ≥ SMA50, ATR-to-SMA50 > 7 (overextension warning)

**Stage 3 - Distribution Phase:**
- 3A (Bullish Fade): Price ≤ EMA10, Price ≤ SMA20, Price ≥ SMA50
- 3B (Fade Confirmation): Price below all MAs

**Stage 4 - Declining Phase:**
- 4A (Bearish Trend): Price below all MAs, ATR-to-SMA50 ≤ 4
- 4B (Bearish Confirmation): Price below all MAs with bearish MA stacking
- 4C (Bearish Extended): Price ≤ SMA50, ATR-to-SMA50 < 4

## Technical Implementation Notes

### Key Conditions
- For Stage 2 classification: Stock must be above SMA50
- Moving average stacking (EMA10 > SMA20 > SMA50) indicates strong bullish momentum
- Inverse stacking (EMA10 < SMA20 < SMA50) indicates strong bearish momentum

### Data Requirements
When implementing or testing this system, ensure access to:
- Daily OHLC price data
- Volume data (recommended for confirmation)
- ATR calculation capability (typically 14-day period)
- Moving average calculations (EMA10, SMA20, SMA50)

## Development Considerations

### File Structure
The repository appears to be part of a larger TradingView indicators collection under `/myIndicators/stage_analysis/`. This suggests the codebase may contain:
- Pine Script indicator files
- Backtesting scripts
- Documentation and examples

### Performance Optimization
- ATR calculations should be optimized for real-time analysis
- Consider implementing caching for moving average calculations
- Stage transitions should be handled efficiently to avoid classification gaps

## References

The system is based on traditional stage analysis concepts (Stan Weinstein's model) with specific adaptations for shorter timeframes and volatility-based refinements. The visual framework maps to Oliver Kell's "Cycle of Price Action" concepts.

## Data Sources

This analysis framework is designed for financial market data and should be implemented with proper market data feeds. Ensure compliance with data provider terms of service and market regulations.