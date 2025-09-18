# Load the TradingView universe data to get industry information
universe_data = pd.read_csv('tradingview_universe.csv')

# Display the structure
print("TradingView Universe Data Structure:")
print(universe_data.columns.tolist())
print("\nFirst few rows:")
print(universe_data.head())

# Check if we need to merge by ticker symbol
print("\nUnique sectors in universe data:")
print(universe_data['Sector'].value_counts())