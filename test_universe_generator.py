#!/usr/bin/env python3
"""
Test script for UniverseGenerator functionality
"""

import sys
import os
from pathlib import Path
import logging

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from config import Config
from user_defined_data import UserConfiguration
from universe_generator import UniverseGenerator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_universe_generator():
    """Test the UniverseGenerator functionality"""
    
    try:
        # Initialize configuration
        config = Config()
        user_config = UserConfiguration()
        
        # Initialize UniverseGenerator
        universe_generator = UniverseGenerator(config, user_config)
        
        print("=== Universe Generator Test ===")
        print(f"Universes directory: {universe_generator.universes_dir}")
        
        # Test 1: Detect required universes
        print("\n1. Detecting required universes...")
        required_universes = universe_generator.detect_required_universes()
        print(f"Required universes: {required_universes}")
        
        # Test 2: Generate specific universe (NASDAQ 100)
        print("\n2. Generating NASDAQ 100 universe (ID: 2)...")
        try:
            nasdaq100_file = universe_generator.generate_universe_file('2')
            print(f"Generated NASDAQ 100 universe: {nasdaq100_file}")
            
            # Check file content
            if nasdaq100_file.exists():
                import pandas as pd
                df = pd.read_csv(nasdaq100_file)
                print(f"NASDAQ 100 universe contains {len(df)} stocks")
                print(f"Sample tickers: {df['ticker'].head().tolist()}")
            
        except Exception as e:
            print(f"Error generating NASDAQ 100 universe: {e}")
        
        # Test 3: Generate all missing universes
        print("\n3. Generating all missing universes...")
        universe_results = universe_generator.generate_missing_universes(required_universes)
        
        print(f"Generated {len(universe_results)} universes:")
        for universe_id, file_path in universe_results.items():
            if file_path.exists():
                import pandas as pd
                df = pd.read_csv(file_path)
                print(f"  Universe {universe_id}: {file_path} ({len(df)} stocks)")
            else:
                print(f"  Universe {universe_id}: {file_path} (FILE NOT FOUND)")
        
        # Test 4: Check metadata
        print("\n4. Checking metadata...")
        metadata_file = universe_generator.metadata_file
        if metadata_file.exists():
            import json
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            print(f"Metadata file: {metadata_file}")
            for universe_id, meta in metadata.items():
                print(f"  Universe {universe_id}: {meta.get('record_count', 0)} stocks, "
                      f"status: {meta.get('validation_status', 'unknown')}")
        else:
            print("No metadata file found")
        
        # Test 5: List generated files
        print("\n5. Generated files:")
        if universe_generator.universes_dir.exists():
            for file_path in universe_generator.universes_dir.glob('*.csv'):
                size_kb = file_path.stat().st_size / 1024
                print(f"  {file_path.name} ({size_kb:.1f} KB)")
        
        print("\n=== Test completed successfully! ===")
        return True
        
    except Exception as e:
        print(f"\n=== Test failed with error: {e} ===")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_universe_generator()
    sys.exit(0 if success else 1)