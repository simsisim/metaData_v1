# ==============================================================================
# Google Colab Stock Data Downloader (to Google Sheets)
# ==============================================================================
#
# INSTRUCTIONS:
# 1. Upload your 'djindexes_googlesheets.csv' file to the main directory of your 
#    Google Drive.
# 2. Open a new notebook in Google Colab (https://colab.research.google.com/).
# 3. Copy and paste the code from each cell below into a new cell in your notebook.
# 4. Run the cells in order. You will be asked to authorize access.

# ==============================================================================
# CELL 1: Install necessary libraries
# ==============================================================================
# We need 'yfinance' to download stock data, 'gspread' to interact with Google
# Sheets, and 'gspread-dataframe' to easily convert our data into a sheet.

!pip install yfinance pandas gspread gspread-dataframe google-auth-oauthlib

# ==============================================================================
# CELL 2: Mount Google Drive and Authenticate for Google APIs
# ==============================================================================
# This cell connects to your Google Drive and authenticates your user account
# so the script has permission to create folders and write new Google Sheets.

from google.colab import auth, drive
from google.auth import default
import gspread

# Authenticate the user for the session. This will allow access to both Sheets and Drive.
auth.authenticate_user()
creds, _ = default()

# Authorize the gspread client.
gc = gspread.authorize(creds)

# Mount Google Drive to access the ticker list file.
drive.mount('/content/drive')


# ==============================================================================
# CELL 3: Main script to download data and create Google Sheets
# ==============================================================================

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from gspread_dataframe import set_with_dataframe

# --- Configuration ---
# Set the path to your input CSV file in Google Drive.
input_csv_path = '/content/drive/MyDrive/_invest2024/DJIndexes_data/djindexes_googlesheets.csv'

# Set the desired output folder path in your Google Drive.
# The script will create this folder structure if it doesn't exist.
output_folder_path = '_invest2024/DJIndexes_data/data/daily'

# IMPORTANT: Set this to your email address so the new sheets are shared with you.
USER_EMAIL = 'your-email@gmail.com' 

# --- Helper Function to get or create folder ---
def get_or_create_folder(folder_path):
    """Finds a folder by path, creating it if it doesn't exist. Returns folder ID."""
    parent_id = 'root'
    # Split path into individual folder names
    folders = folder_path.split('/')
    for folder_name in folders:
        # Query for the folder within the current parent
        query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and '{parent_id}' in parents and trashed=false"
        results = gc.drive.list_file_info(q=query).get('files')
        
        if results:
            # Folder found, it becomes the new parent
            parent_id = results[0]['id']
        else:
            # Folder not found, create it
            print(f"Creating folder: {folder_name}")
            folder_metadata = {
                'name': folder_name,
                'mimeType': 'application/vnd.google-apps.folder',
                'parents': [parent_id]
            }
            folder = gc.drive.create_file(metadata=folder_metadata)
            folder.upload()
            parent_id = folder['id']
    return parent_id

# --- Main Logic ---

# 1. Get the destination folder ID, creating folders as needed
print(f"Ensuring output folder exists: '{output_folder_path}'")
destination_folder_id = get_or_create_folder(output_folder_path)
print(f"All sheets will be saved in folder ID: {destination_folder_id}")

# 2. Read the list of tickers from your CSV file
try:
    print(f"\nReading tickers from: {input_csv_path}")
    df_tickers = pd.read_csv(input_csv_path, header=None)
    tickers = df_tickers[0].tolist()
    print(f"Successfully found {len(tickers)} tickers.")
except FileNotFoundError:
    print(f"ERROR: File not found at '{input_csv_path}'.")
    print("Please make sure you have uploaded 'djindexes_googlesheets.csv' to your Google Drive.")
else:
    # 3. Define the date range (last 1 year)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    # 4. Loop through each ticker, download data, and save to a new Google Sheet
    print("\nStarting data download process...")
    for ticker in tickers:
        sanitized_ticker = ticker.replace('^', '')
        
        try:
            print(f"  -> Processing ticker: {ticker}...")
            stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)

            if stock_data.empty:
                print(f"     - WARNING: No data found for {ticker}.")
                continue

            stock_data.reset_index(inplace=True)

            print(f"     - Creating Google Sheet named '{sanitized_ticker}'...")
            spreadsheet = gc.create(sanitized_ticker, folder_id=destination_folder_id)
            worksheet = spreadsheet.sheet1
            set_with_dataframe(worksheet, stock_data)
            
            # The share step is still useful to ensure you have direct editor access
            spreadsheet.share(USER_EMAIL, perm_type='user', role='writer')
            
            print(f"     - Success! Sheet created in target folder. Link: {spreadsheet.url}")

        except Exception as e:
            print(f"     - ERROR: Failed to process data for {ticker}. Reason: {e}")

    print("\nScript finished. All data has been saved to your specified Google Drive folder.")

