#!/usr/bin/env python3
"""
Sparkline HTML Generator
Creates HTML report with embedded sparkline charts using processed ticker data
"""

import os
import sys
import json
from datetime import datetime

class SparklineHTMLGenerator:
    """Generate HTML report with sparkline charts"""

    def __init__(self, project_dir=None):
        if project_dir is None:
            self.project_dir = os.path.dirname(os.path.dirname(__file__))
        else:
            self.project_dir = project_dir

        self.template_path = os.path.join(self.project_dir, 'templates', 'sparkline_report_template.html')
        self.outputs_dir = os.path.join(self.project_dir, 'outputs')

        # Ensure outputs directory exists
        os.makedirs(self.outputs_dir, exist_ok=True)

    def load_template(self):
        """Load the HTML template"""
        try:
            with open(self.template_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"Template not found: {self.template_path}")

    def load_processed_data(self, data_file="processed_sparkline_data.json"):
        """Load processed sparkline data"""
        data_path = os.path.join(self.outputs_dir, data_file)

        try:
            with open(data_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Processed data not found: {data_path}")

    def format_price_data(self, prices):
        """Format price data for JavaScript sparkline"""
        if not prices:
            return "[]"

        # Convert to comma-separated string
        formatted_prices = ",".join(str(round(price, 2)) for price in prices)
        return f"[{formatted_prices}]"

    def get_trend_badge_class(self, trend):
        """Get CSS class for trend badge"""
        trend_classes = {
            'bullish': 'badge-bullish',
            'bearish': 'badge-bearish',
            'neutral': 'badge-neutral'
        }
        return trend_classes.get(trend, 'badge-neutral')

    def generate_ticker_row(self, ticker_data):
        """Generate HTML table row for a ticker"""
        ticker = ticker_data['ticker']
        current_price = ticker_data.get('current_price', 0)
        timeframes = ticker_data.get('timeframes', {})

        # Start row
        row_html = f'<tr data-ticker="{ticker}">\n'

        # Ticker name column
        row_html += f'    <td class="ticker-name">{ticker}</td>\n'

        # Timeframe columns
        timeframe_order = ['1d', '1w', '1m', '3m', '6m', '1y']

        for timeframe in timeframe_order:
            timeframe_data = timeframes.get(timeframe, {})
            prices = timeframe_data.get('prices', [])
            metrics = timeframe_data.get('metrics', {})

            total_return = metrics.get('total_return', 0)
            trend = metrics.get('trend', 'neutral')
            data_points = timeframe_data.get('data_points', 0)

            # Format price data for sparkline
            sparkline_data = self.format_price_data(prices)

            # Determine performance color
            perf_color = 'text-success' if total_return > 0 else 'text-danger' if total_return < 0 else 'text-muted'
            perf_sign = '+' if total_return > 0 else ''

            # Generate cell HTML - performance badge first, then chart container
            row_html += f'''    <td class="sparkline-cell">
        <div class="performance-badge {self.get_trend_badge_class(trend)}">
            {perf_sign}{total_return:.1f}%
        </div>
        <div class="sparkline-container">
            <span class="sparkline"
                  data-values="{sparkline_data}"
                  data-timeframe="{timeframe}"
                  data-performance="{total_return}"
                  data-bs-toggle="tooltip"
                  title="{timeframe.upper()}: {perf_sign}{total_return:.1f}% ({data_points} days)">
            </span>
        </div>
    </td>\n'''

        # Close row
        row_html += '</tr>\n'

        return row_html

    def calculate_summary_stats(self, all_ticker_data):
        """Calculate summary statistics for the header"""
        total_tickers = len(all_ticker_data)

        # Count bullish/bearish based on 1-year performance
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0

        for ticker_data in all_ticker_data:
            year_metrics = ticker_data.get('timeframes', {}).get('1y', {}).get('metrics', {})
            trend = year_metrics.get('trend', 'neutral')

            if trend == 'bullish':
                bullish_count += 1
            elif trend == 'bearish':
                bearish_count += 1
            else:
                neutral_count += 1

        # Find most recent update date
        last_updated = "N/A"
        if all_ticker_data:
            dates = [td.get('last_update', '') for td in all_ticker_data if td.get('last_update')]
            if dates:
                last_updated = max(dates)

        return {
            'total_tickers': total_tickers,
            'bullish_count': bullish_count,
            'bearish_count': bearish_count,
            'neutral_count': neutral_count,
            'last_updated': last_updated
        }

    def generate_html_report(self, data_file="processed_sparkline_data.json"):
        """Generate complete HTML report"""
        print("📄 Loading HTML template...")
        template = self.load_template()

        print("📊 Loading processed data...")
        all_ticker_data = self.load_processed_data(data_file)

        print(f"📈 Generating HTML for {len(all_ticker_data)} tickers...")

        # Generate table rows
        table_rows = ""
        for ticker_data in all_ticker_data:
            row = self.generate_ticker_row(ticker_data)
            table_rows += row

        # Calculate summary statistics
        stats = self.calculate_summary_stats(all_ticker_data)

        # Template substitutions
        substitutions = {
            '{{TABLE_ROWS}}': table_rows,
            '{{TOTAL_TICKERS}}': str(stats['total_tickers']),
            '{{BULLISH_COUNT}}': str(stats['bullish_count']),
            '{{BEARISH_COUNT}}': str(stats['bearish_count']),
            '{{LAST_UPDATED}}': stats['last_updated'],
            '{{GENERATION_DATE}}': datetime.now().strftime('%B %d, %Y at %I:%M %p')
        }

        # Apply substitutions
        final_html = template
        for placeholder, value in substitutions.items():
            final_html = final_html.replace(placeholder, value)

        return final_html

    def save_html_report(self, html_content, filename=None):
        """Save HTML report to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"historical_price_sparklines_{timestamp}.html"

        output_path = os.path.join(self.outputs_dir, filename)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        # Get file size
        file_size = os.path.getsize(output_path) / 1024  # KB

        print(f"✅ HTML report saved: {output_path}")
        print(f"📄 File size: {file_size:.1f} KB")

        return output_path

    def create_complete_report(self, data_file="processed_sparkline_data.json"):
        """Create complete sparkline HTML report"""
        print("\n" + "="*60)
        print("🚀 Starting HTML Report Generation")
        print("="*60)

        try:
            # Generate HTML
            html_content = self.generate_html_report(data_file)

            # Save report
            output_path = self.save_html_report(html_content)

            print("\n📊 Report Summary:")
            print(f"   • Template: sparkline_report_template.html")
            print(f"   • Data source: {data_file}")
            print(f"   • Output: {os.path.basename(output_path)}")
            print(f"   • Interactive features: Search, Sort, Tooltips, Export")

            print("\n✅ HTML Report Generation Complete!")
            print("="*60)

            return output_path

        except Exception as e:
            print(f"❌ Error generating HTML report: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """Main execution function"""
    try:
        generator = SparklineHTMLGenerator()
        output_file = generator.create_complete_report()

        if output_file:
            print(f"\n🎉 Success! Open the report in your browser:")
            print(f"file://{output_file}")
        else:
            print("❌ Failed to generate HTML report")

    except Exception as e:
        print(f"❌ Error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()