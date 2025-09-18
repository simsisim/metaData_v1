# Perfect! All trading data tickers are in the universe data. Let's merge the datasets
# Merge the datasets
merged_data = trading_data.merge(universe_data, left_on='ticker', right_on='Symbol', how='left')

print(f"Merged data shape: {merged_data.shape}")
print("Merge successful - no missing matches!")

# Clean and parse the Index column
def parse_indices(index_str):
    """Parse the Index column to extract individual indices"""
    if pd.isna(index_str):
        return []
    return [idx.strip() for idx in str(index_str).split(',')]

# Extract individual indices for each stock
merged_data['indices_list'] = merged_data['Index'].apply(parse_indices)

# Create binary columns for major indices
major_indices = ['S&P 500', 'NASDAQ 100', 'NASDAQ Composite', 'Dow Jones Industrial Average', 'Russell 1000', 'Russell 3000']

for index in major_indices:
    merged_data[f'in_{index.replace(" ", "_").replace("&", "").lower()}'] = merged_data['indices_list'].apply(
        lambda x: index in x if x else False
    )

print("\nMajor indices representation:")
for index in major_indices:
    col_name = f'in_{index.replace(" ", "_").replace("&", "").lower()}'
    count = merged_data[col_name].sum()
    print(f"{index}: {count} stocks ({count/len(merged_data)*100:.1f}%)")