import os
import shutil
from datetime import datetime
from pathlib import Path

class Config:
    def __init__(self, user_choice=None):
        self.current_date = datetime.now().strftime('%Y%m%d')
        self.user_choice = user_choice or "default"
        # Get the directory of the current script
        self.base_dir = Path(__file__).resolve().parent

        # Source and destination paths (relative to project root)
        self.paths = {
            'source_market_data': str(self.base_dir.parent.parent / "downloadData_v1" / "data" / "market_data/daily/"),
            'source_tickers_data': str(self.base_dir.parent.parent / "downloadData_v1" / "data" / "tickers"),
            'dest_tickers_data': str(self.base_dir.parent / "scanners" / "tickers"),
            'info_tickers_static': str(self.base_dir / "info_tickers_static.csv")
        }
        
        # Directories (relative to project root)
        self.directories = {
            'results': str(self.base_dir / "scanners" / "results"),
            'tickers': str(self.base_dir / "scanners" / "tickers")
        }
        
        self.market_data_params = {
            'source_market_data': str(self.base_dir.parent.parent / "downloadData_v1" / "data" / "market_data"),
            'results_dir': str(self.base_dir.parent / "scanners" / "results")
        }
        
        # Results files
        self.results_files = {
            'PVB_filename': None,
            'ATR_filename': None,
            'darvas_filename': None
        }
        
        self.pvb_params = {
            'price_breakout_period': 20,
            'volume_breakout_period': 20,
            'trendline_length': 50,
            'order_direction': "Long and Short",
            'PVB_dir': str(self.base_dir.parent / "scanners" / "results" / "PVB")
        }

        self.atr_params = {
            'vstop_factor': 3,
            'vstop_length': 20,
            'vstop_factor2': 1.5,
            'vstop_length2': 20,
            'src': 'Close',
            'src2': 'Close',
            'ATR_dir': str(self.base_dir.parent / "scanners" / "results" / "ATR")
        }
        
        self.darvas_params = {
            'box_length': 5,
            'darvas_dir': str(self.base_dir.parent / "scanners" / "results" / "darvas")
        }
        
        self.BOR_params = {
            'lookback': 20,
            'bars_since_breakout': 2,
            'retest_limiter': 2,
            'BOR_dir': str(self.base_dir.parent / "scanners" / "results" / "BOR")
        }
        
        self.BOSR_params = {
            'pivot_no': 5,  # 5 before; 5 after
            'BOSR_dir': str(self.base_dir.parent / "scanners" / "results" / "BOSR")
        }
        
        self.create_directory_structure()
        self.copy_folder()

    def create_directory_structure(self):
        os.makedirs(self.pvb_params['PVB_dir'], exist_ok=True)
        os.makedirs(self.atr_params['ATR_dir'], exist_ok=True)
        os.makedirs(self.darvas_params['darvas_dir'], exist_ok=True)
        os.makedirs(self.BOR_params['BOR_dir'], exist_ok=True)
        os.makedirs(self.BOSR_params['BOSR_dir'], exist_ok=True)


    def copy_folder(self):
        source = self.paths['source_tickers_data']
        dest = self.paths['dest_tickers_data']
        try:
            shutil.copytree(source, dest, dirs_exist_ok=True)
        except FileNotFoundError as e:
            raise RuntimeError(f"Path not found: {e.filename}.\n"
                              f"Make sure 'downloadData_v1' exists at: {Path(source).parent.parent}") from e

    def update_params(self, user_choice):
        self.user_choice = user_choice
        self.current_date = datetime.now().strftime('%Y%m%d')
        results_dir = self.market_data_params['results_dir']
        self.results_files['PVB_filename'] = os.path.join(results_dir, f'PVB_{self.user_choice}_{self.current_date}.csv')
        self.results_files['ATR_filename'] = os.path.join(results_dir, f'ATR_{self.user_choice}_{self.current_date}.csv')
        self.results_files['darvas_filename'] = os.path.join(results_dir, f'darvas_{self.user_choice}_{self.current_date}.csv')
        # If you want to use BOR and BOSR in results_files, add them here:
        # self.results_files['BOR_filename'] = os.path.join(results_dir, f'BOR_{self.user_choice}_{self.current_date}.csv')
        # self.results_files['BOSR_filename'] = os.path.join(results_dir, f'BOSR_{self.user_choice}_{self.current_date}.csv')
        self.pvb_params['PVB_filename'] = self.results_files['PVB_filename']
        self.atr_params['ATR_filename'] = self.results_files['ATR_filename']
        self.darvas_params['darvas_filename'] = self.results_files['darvas_filename']
        # If you want to set BOR and BOSR filenames in their params, add:
        self.BOR_params['BOR_filename'] = os.path.join(results_dir, f'BOR_{self.user_choice}_{self.current_date}.csv')
        self.BOSR_params['BOSR_filename'] = os.path.join(results_dir, f'BOSR_{self.user_choice}_{self.current_date}.csv')

