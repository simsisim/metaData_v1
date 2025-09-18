# Sector Analysis across timeframes
print("SECTOR ANALYSIS ACROSS TIMEFRAMES")
print("="*80)

# Short-term sector performance
print("\nSHORT-TERM SECTOR PERFORMANCE:")
st_sector = short_term_results.groupby('sector').agg({
    'short_term_score': 'mean',
    '5d_change': 'mean',
    'ticker': 'count'
}).round(2)
st_sector.columns = ['avg_st_score', 'avg_5d_change', 'count']
st_sector = st_sector.sort_values('avg_st_score', ascending=False)
print(st_sector)

# Medium-term sector performance  
print("\nMEDIUM-TERM SECTOR PERFORMANCE:")
mt_sector = medium_term_results.groupby('sector').agg({
    'medium_term_score': 'mean',
    '44d_change': 'mean',
    'ticker': 'count'
}).round(2)
mt_sector.columns = ['avg_mt_score', 'avg_44d_change', 'count']
mt_sector = mt_sector.sort_values('avg_mt_score', ascending=False)
print(mt_sector)

# Long-term sector performance
print("\nLONG-TERM SECTOR PERFORMANCE:")
lt_sector = long_term_results.groupby('sector').agg({
    'long_term_score': 'mean', 
    '252d_change': 'mean',
    'ticker': 'count'
}).round(2)
lt_sector.columns = ['avg_lt_score', 'avg_252d_change', 'count']
lt_sector = lt_sector.sort_values('avg_lt_score', ascending=False)
print(lt_sector)