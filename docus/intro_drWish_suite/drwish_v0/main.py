import pandas as pd
from src.data_reader import DataReader
from src.tickers_choice import get_ticker_files, user_choice
from src.combined_tickers import combine_tickers
from src.glb_calculator import GLBCalculator, load_glb_config
from src.glb_charts import GLBChartGenerator
from src.blue_dot_calculator import BlueDotCalculator
from src.black_dot_calculator import BlackDotCalculator
#from src.extended_results import extend_rs_results 
#from src.basic_calculation import basic_calculation

#from src.price_volume_breakout import run_PVBstrategy
#from src.atr_cloud import apply_atr_cloud, run_atr_cloud_strategy
#from src.darvas import run_darvas_strategy
#from src.pivot_points import run_breakout_strategy
#from src.supp_resist_breakout import run_supp_resist_breakout_strategy#, print_supp_resist_breakout_results


from src.config import Config
import os 
import logging
from datetime import datetime

#logging.basicConfig(level=logging.INFO)

def main():
    batch_size = 10

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.getLogger().setLevel(logging.CRITICAL)

    # Load GLB configuration
    print("🔧 Loading GLB Configuration...")
    glb_config = load_glb_config('glb_config.csv')
    print(f"   Pivot Strength: {glb_config.get('pivot_strength')}")
    print(f"   Timeframe: {glb_config.get('timeframe_mode')}")
    print(f"   Lookback Period: {glb_config.get('lookback_period')}")
    print(f"   Confirmation Period: {glb_config.get('confirmation_period')}")
    print(f"   Require Confirmation: {glb_config.get('require_confirmation')}")
    print(f"   Show Blue Dots: {glb_config.get('show_blue_dots')}")
    print(f"   Blue Dot Stoch Period: {glb_config.get('blue_dot_stoch_period')}")
    print(f"   Blue Dot Stoch Threshold: {glb_config.get('blue_dot_stoch_threshold')}")
    print(f"   Show Black Dots: {glb_config.get('show_black_dots')}")
    print(f"   Black Dot Stoch Threshold: {glb_config.get('black_dot_stoch_threshold')}")
    print(f"   Black Dot Lookback: {glb_config.get('black_dot_lookback')}")

    # Print initial configuration
    ticker_files, user_choice = get_ticker_files()
    print(f"\n🔧 Ticker Configuration:")
    print(f"   User choice: {user_choice}")
    print(f"   Ticker files to load: {ticker_files}")
    
    config = Config(user_choice)
    config.update_params(user_choice)

    # Combine tickers and show results
    combined_file, info_tickers_file_path, industry_df, sector_df, industry_names, sector_names = combine_tickers(ticker_files, config.paths)
    print(f"\n📁 Generated files:")
    print(f"   Combined tickers: {combined_file}")
    print(f"   Info tickers: {info_tickers_file_path}")
    
    # Load and display the actual tickers that will be processed
    if os.path.exists(combined_file):
        tickers_df = pd.read_csv(combined_file)
        ticker_list = tickers_df['ticker'].tolist()
        print(f"\n📊 Tickers to process ({len(ticker_list)} total):")
        print(f"   {ticker_list}")
    else:
        print(f"❌ Combined file not found: {combined_file}")
        return
    
    # Create output directories
    output_path = os.path.join(config.paths['dest_tickers_data'], 'basic_calculations')
    glb_results_path = os.path.join(config.paths['dest_tickers_data'], 'glb_results')
    glb_charts_path = os.path.join(config.paths['dest_tickers_data'], 'glb_charts')
    
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(glb_results_path, exist_ok=True)
    os.makedirs(glb_charts_path, exist_ok=True)
    
    # Initialize GLB, Blue Dot, and Black Dot components
    glb_calculator = GLBCalculator(glb_config)
    blue_dot_calculator = BlueDotCalculator(glb_config)
    black_dot_calculator = BlackDotCalculator(glb_config)
    chart_generator = GLBChartGenerator(glb_config)
    
    # Initialize DataReader
    data_reader = DataReader(config.paths, combined_file, batch_size)
    print(f"\n⚙️ Processing setup:")
    print(f"   Batch size: {batch_size}")
    print(f"   Data source: {config.paths['source_market_data']}")
    print(f"   GLB Results: {glb_results_path}")
    print(f"   GLB Charts: {glb_charts_path}")

    # Process batches with GLB, Blue Dot, and Black Dot analysis
    batch_number = 1
    total_processed = 0
    all_glb_results = []
    all_blue_dot_results = []
    all_black_dot_results = []
    
    for batch in data_reader.get_batches():
        print(f"\n🔄 Processing Batch {batch_number}:")
        print(f"   Tickers in this batch: {batch}")
        
        # Read CSV data for the batch
        batch_data = {}
        successful_loads = []
        failed_loads = []
        
        for ticker in batch:
            try:
                data = data_reader.read_stock_data(ticker)
                if data is not None and len(data) > 0:
                    batch_data[ticker] = data
                    successful_loads.append(ticker)
                    print(f"   ✅ {ticker}: {len(data)} rows loaded")
                else:
                    failed_loads.append(ticker)
                    print(f"   ⚠️  {ticker}: No data available")
            except Exception as e:
                failed_loads.append(ticker)
                print(f"   ❌ {ticker}: Error loading - {str(e)}")
        
        print(f"   📊 Batch {batch_number} summary: {len(successful_loads)} successful, {len(failed_loads)} failed")
        
        # Perform GLB, Blue Dot, and Black Dot analysis on successful loads
        if batch_data:
            print(f"\n📈 Performing GLB, Blue Dot & Black Dot Analysis for Batch {batch_number}:")
            
            for ticker, data in batch_data.items():
                try:
                    # Convert DataReader format to calculator format
                    data_converted = data.copy()
                    data_converted.reset_index(inplace=True)
                    
                    # Map column names to calculator expected format
                    column_mapping = {
                        'Date': 'date',
                        'Close': 'close', 
                        'High': 'high',
                        'Low': 'low',
                        'Volume': 'volume'
                    }
                    data_converted = data_converted.rename(columns=column_mapping)
                    
                    # Add missing 'open' column (use previous close as approximation)
                    data_converted['open'] = data_converted['close'].shift(1)
                    data_converted['open'].fillna(data_converted['close'], inplace=True)
                    
                    # Add ticker column
                    data_converted['ticker'] = ticker
                    
                    # Calculate GLB signals
                    glb_results = glb_calculator.calculate_glb_signals(data_converted)
                    all_glb_results.append(glb_results)
                    
                    # Calculate Blue Dot signals
                    blue_dot_results = blue_dot_calculator.detect_blue_dot_signals(data_converted)
                    all_blue_dot_results.append(blue_dot_results)
                    
                    # Calculate Black Dot signals
                    black_dot_results = black_dot_calculator.detect_black_dot_signals(data_converted)
                    all_black_dot_results.append(black_dot_results)
                    
                    # Print analysis statistics
                    stats = glb_results['statistics']
                    current_glb = glb_results['current_glb']
                    blue_stats = blue_dot_results['statistics']
                    blue_current = blue_dot_results['current_conditions']
                    black_stats = black_dot_results['statistics']
                    black_current = black_dot_results['current_conditions']
                    
                    print(f"   📊 {ticker} Analysis:")
                    print(f"      GLB - Total: {stats['total_glbs']}, Confirmed: {stats['confirmed_glbs']}, Broken: {stats['broken_glbs']}")
                    print(f"      GLB - Current: {current_glb['level']:.2f}" if current_glb['active'] else "      GLB - Current: None")
                    print(f"      Blue Dot - Total: {blue_stats['total_signals']}, Recent: {'YES' if blue_stats['recent_signal'] else 'NO'}")
                    print(f"      Black Dot - Total: {black_stats['total_signals']}, Ready: {'YES' if black_current['ready_for_signal'] else 'NO'}")
                    
                    # Generate and save chart with Blue/Black Dot integration
                    chart_path = os.path.join(glb_charts_path, f"{ticker}_glb_complete_chart.png")
                    try:
                        fig = chart_generator.create_glb_chart(data_converted, glb_results, ticker, blue_dot_results, black_dot_results, chart_path)
                        print(f"      Chart saved: {chart_path}")
                        import matplotlib.pyplot as plt
                        plt.close(fig)  # Free memory
                    except Exception as chart_error:
                        print(f"      ⚠️ Chart generation failed: {chart_error}")
                    
                    # Save GLB results to CSV
                    glb_results_file = os.path.join(glb_results_path, f"{ticker}_glb_results.csv")
                    try:
                        if glb_results['glb_records']:
                            results_df = pd.DataFrame(glb_results['glb_records'])
                            results_df.to_csv(glb_results_file, index=False)
                            print(f"      GLB Results saved: {glb_results_file}")
                    except Exception as save_error:
                        print(f"      ⚠️ GLB Results save failed: {save_error}")
                    
                    # Save Blue Dot results to CSV
                    blue_dot_results_file = os.path.join(glb_results_path, f"{ticker}_blue_dot_results.csv")
                    try:
                        if blue_dot_results['blue_dot_signals']:
                            blue_dot_df = pd.DataFrame(blue_dot_results['blue_dot_signals'])
                            blue_dot_df.to_csv(blue_dot_results_file, index=False)
                            print(f"      Blue Dot Results saved: {blue_dot_results_file}")
                    except Exception as save_error:
                        print(f"      ⚠️ Blue Dot Results save failed: {save_error}")
                    
                    # Save Black Dot results to CSV
                    black_dot_results_file = os.path.join(glb_results_path, f"{ticker}_black_dot_results.csv")
                    try:
                        if black_dot_results['black_dot_signals']:
                            black_dot_df = pd.DataFrame(black_dot_results['black_dot_signals'])
                            black_dot_df.to_csv(black_dot_results_file, index=False)
                            print(f"      Black Dot Results saved: {black_dot_results_file}")
                    except Exception as save_error:
                        print(f"      ⚠️ Black Dot Results save failed: {save_error}")
                        
                except Exception as e:
                    print(f"   ❌ {ticker}: Analysis failed - {str(e)}")
        
        total_processed += len(batch)
        batch_number += 1
    
    # Generate summary report and chart
    if all_glb_results and all_blue_dot_results and all_black_dot_results:
        print(f"\n📊 Generating GLB, Blue Dot & Black Dot Summary Report...")
        
        # Create GLB summary statistics
        total_glbs_all = sum(result['statistics']['total_glbs'] for result in all_glb_results)
        total_broken_all = sum(result['statistics']['broken_glbs'] for result in all_glb_results)
        total_confirmed_all = sum(result['statistics']['confirmed_glbs'] for result in all_glb_results)
        active_glb_count = sum(1 for result in all_glb_results if result['current_glb']['active'])
        
        # Create Blue Dot summary statistics
        total_blue_dots_all = sum(result['statistics']['total_signals'] for result in all_blue_dot_results)
        recent_blue_dot_count = sum(1 for result in all_blue_dot_results if result['statistics']['recent_signal'])
        
        # Create Black Dot summary statistics
        total_black_dots_all = sum(result['statistics']['total_signals'] for result in all_black_dot_results)
        ready_black_dot_count = sum(1 for result in all_black_dot_results if result['current_conditions']['ready_for_signal'])
        
        print(f"   📈 Overall GLB Statistics:")
        print(f"      Total GLBs detected: {total_glbs_all}")
        print(f"      Total GLBs confirmed: {total_confirmed_all}")
        print(f"      Total GLBs broken: {total_broken_all}")
        print(f"      Tickers with active GLBs: {active_glb_count}")
        print(f"      GLB breakout rate: {(total_broken_all/total_glbs_all*100):.1f}%" if total_glbs_all > 0 else "      GLB breakout rate: N/A")
        
        print(f"   🔵 Overall Blue Dot Statistics:")
        print(f"      Total Blue Dots detected: {total_blue_dots_all}")
        print(f"      Tickers with recent Blue Dots: {recent_blue_dot_count}")
        print(f"      Blue Dot frequency: {(total_blue_dots_all/len(all_blue_dot_results)):.1f} per ticker" if all_blue_dot_results else "      Blue Dot frequency: N/A")
        
        print(f"   ⚫ Overall Black Dot Statistics:")
        print(f"      Total Black Dots detected: {total_black_dots_all}")
        print(f"      Tickers ready for Black Dot: {ready_black_dot_count}")
        print(f"      Black Dot frequency: {(total_black_dots_all/len(all_black_dot_results)):.1f} per ticker" if all_black_dot_results else "      Black Dot frequency: N/A")
        
        # Generate summary chart
        summary_chart_path = os.path.join(glb_charts_path, "GLB_Summary_Analysis.png")
        try:
            summary_fig = chart_generator.create_summary_chart(all_glb_results, summary_chart_path)
            print(f"   📊 Summary chart saved: {summary_chart_path}")
            import matplotlib.pyplot as plt
            plt.close(summary_fig)
        except Exception as e:
            print(f"   ⚠️ Summary chart generation failed: {e}")
        
        # Save comprehensive results summary
        summary_file = os.path.join(glb_results_path, "Complete_DrWish_Analysis_Summary.csv")
        try:
            summary_data = []
            for glb_result, blue_dot_result, black_dot_result in zip(all_glb_results, all_blue_dot_results, all_black_dot_results):
                glb_stats = glb_result['statistics']
                current_glb = glb_result['current_glb']
                blue_stats = blue_dot_result['statistics']
                blue_current = blue_dot_result['current_conditions']
                black_stats = black_dot_result['statistics']
                black_current = black_dot_result['current_conditions']
                
                summary_data.append({
                    'ticker': glb_result['ticker'],
                    # GLB data
                    'total_glbs': glb_stats['total_glbs'],
                    'confirmed_glbs': glb_stats['confirmed_glbs'],
                    'broken_glbs': glb_stats['broken_glbs'],
                    'active_glbs': glb_stats['active_glbs'],
                    'current_glb_level': current_glb['level'] if current_glb['active'] else None,
                    'has_active_glb': current_glb['active'],
                    'glb_breakout_rate': (glb_stats['broken_glbs']/glb_stats['total_glbs']*100) if glb_stats['total_glbs'] > 0 else 0,
                    # Blue Dot data
                    'total_blue_dots': blue_stats['total_signals'],
                    'blue_dots_per_year': blue_stats['signals_per_year'],
                    'recent_blue_dot': blue_stats['recent_signal'],
                    'current_stochastic': blue_current['stochastic'],
                    'stoch_below_threshold': blue_current['stoch_below_threshold'],
                    'sma_rising': blue_current['sma_rising'],
                    # Black Dot data
                    'total_black_dots': black_stats['total_signals'],
                    'black_dots_per_year': black_stats['signals_per_year'],
                    'recent_black_dot': black_stats['recent_signal'],
                    'ready_for_black_dot': black_current['ready_for_signal'],
                    'above_sma': black_current['above_sma'],
                    'above_ema': black_current['above_ema'],
                    'price_rising': black_current['price_rising'],
                    # General data
                    'analysis_date': glb_result['analysis_date'].strftime('%Y-%m-%d %H:%M:%S'),
                    'data_start': glb_result['data_period']['start_date'],
                    'data_end': glb_result['data_period']['end_date'],
                    'total_bars': glb_result['data_period']['total_bars']
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(summary_file, index=False)
            print(f"   📄 Summary results saved: {summary_file}")
            
        except Exception as e:
            print(f"   ⚠️ Summary save failed: {e}")

    # Print final summary
    print(f"\n✅ Dr. Wish Complete Strategy Analysis Complete!")
    print(f"   Total tickers processed: {total_processed}")
    print(f"   Total batches: {batch_number - 1}")
    print(f"   GLB Results directory: {glb_results_path}")
    print(f"   GLB Charts directory: {glb_charts_path}")
    print(f"   Configuration used: glb_config.csv")
    print(f"\n🎯 Ready for complete Dr. Wish 3-strategy implementation!")
    
    # Display tickers with active GLBs for immediate attention
    if all_glb_results:
        active_glb_tickers = [result['ticker'] for result in all_glb_results if result['current_glb']['active']]
        if active_glb_tickers:
            print(f"\n🚨 Tickers with ACTIVE GLBs (Ready for Breakout):")
            for ticker in active_glb_tickers:
                result = next(r for r in all_glb_results if r['ticker'] == ticker)
                glb_level = result['current_glb']['level']
                print(f"   📈 {ticker}: GLB at ${glb_level:.2f}")
        else:
            print(f"\n📝 No active GLBs currently detected in processed tickers.")
    
    # Display tickers with recent Blue Dot signals
    if all_blue_dot_results:
        recent_blue_dot_tickers = [result['ticker'] for result in all_blue_dot_results if result['statistics']['recent_signal']]
        if recent_blue_dot_tickers:
            print(f"\n🔵 Tickers with RECENT Blue Dot Signals (Oversold Bounce Opportunities):")
            for ticker in recent_blue_dot_tickers:
                result = next(r for r in all_blue_dot_results if r['ticker'] == ticker)
                stoch_value = result['current_conditions']['stochastic']
                signals_count = result['statistics']['total_signals']
                print(f"   🔵 {ticker}: {signals_count} Blue Dots, Current Stoch: {stoch_value:.1f}" if stoch_value else f"   🔵 {ticker}: {signals_count} Blue Dots")
        else:
            print(f"\n📝 No recent Blue Dot signals detected in processed tickers.")
    
    # Display tickers ready for Black Dot signals
    if all_black_dot_results:
        ready_black_dot_tickers = [result['ticker'] for result in all_black_dot_results if result['current_conditions']['ready_for_signal']]
        if ready_black_dot_tickers:
            print(f"\n⚫ Tickers READY for Black Dot Signals (Oversold + Above MAs):")
            for ticker in ready_black_dot_tickers:
                result = next(r for r in all_black_dot_results if r['ticker'] == ticker)
                stoch_value = result['current_conditions']['stochastic']
                signals_count = result['statistics']['total_signals']
                above_sma = result['current_conditions']['above_sma']
                above_ema = result['current_conditions']['above_ema']
                ma_status = "SMA+EMA" if above_sma and above_ema else "SMA" if above_sma else "EMA" if above_ema else "None"
                print(f"   ⚫ {ticker}: {signals_count} Black Dots, Stoch: {stoch_value:.1f}, Above: {ma_status}" if stoch_value else f"   ⚫ {ticker}: {signals_count} Black Dots, Above: {ma_status}")
        else:
            print(f"\n📝 No tickers currently ready for Black Dot signals.")



if __name__ == "__main__":
    main()
#### CALL FUNCTION TO PERFORM CALCULATION

