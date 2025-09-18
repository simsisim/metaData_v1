# Load and examine the relative strength (RS) data
rs_daily = pd.read_csv('rs_ibd_stocks_daily_0_20250829.csv')
rs_weekly = pd.read_csv('rs_ibd_stocks_weekly_0_20250825.csv')
rs_monthly = pd.read_csv('rs_ibd_stocks_monthly_0_20250829.csv')

print("--- Relative Strength Data ---")
print("RS Daily Shape:", rs_daily.shape)
print("RS Weekly Shape:", rs_weekly.shape)
print("RS Monthly Shape:", rs_monthly.shape)

# Load stage analysis data
stage_daily = pd.read_csv('stage_analysis_daily_0_20250829.csv')
stage_weekly = pd.read_csv('stage_analysis_weekly_0_20250825.csv')
stage_monthly = pd.read_csv('stage_analysis_monthly_0_20250829.csv')

print("\n--- Stage Analysis Data ---")
print("Stage Daily Shape:", stage_daily.shape)
print("Stage Weekly Shape:", stage_weekly.shape)
print("Stage Monthly Shape:", stage_monthly.shape)

# Load sector and industry RS data
rs_sectors_daily = pd.read_csv('rs_ibd_sectors_daily_0_20250906.csv')
rs_industries_daily = pd.read_csv('rs_ibd_industries_daily_0_20250906.csv')

print("\n--- Sector/Industry RS Data ---")
print("Sectors Daily Shape:", rs_sectors_daily.shape)
print("Industries Daily Shape:", rs_industries_daily.shape)

print("\n--- Sample of Sector RS Data ---")
print(rs_sectors_daily)