# Create index membership analysis data
index_analysis = []

for index in major_indices:
    col_name = f'in_{index.replace(" ", "_").replace("&", "").lower()}'
    in_index = merged_data[merged_data[col_name] == True]
    not_in_index = merged_data[merged_data[col_name] == False]
    
    index_analysis.append({
        'Index': index,
        'In_Index_Count': len(in_index),
        'In_Index_1Y_Return_Avg': in_index['daily_daily_yearly_252d_pct_change'].mean(),
        'Not_In_Index_1Y_Return_Avg': not_in_index['daily_daily_yearly_252d_pct_change'].mean(),
        'In_Index_1D_Return_Avg': in_index['daily_daily_daily_1d_pct_change'].mean(),
        'Not_In_Index_1D_Return_Avg': not_in_index['daily_daily_daily_1d_pct_change'].mean()
    })

index_analysis_df = pd.DataFrame(index_analysis)
index_analysis_df.to_csv('index_analysis.csv', index=False)
print("Index analysis data created:")
print(index_analysis_df.round(2))