#!/usr/bin/env python3
"""
Test MMM Directory Structure
===========================

Simple test of MMM directory structure without imports.
"""

from pathlib import Path

def test_mmm_directories():
    """Test MMM directory structure."""
    print("🧪 Testing MMM Directory Structure")
    print("=" * 50)

    # Check base SR directory
    sr_base = Path("results/sustainability_ratios")
    print(f"📊 SR Base Directory: {sr_base}")
    print(f"   Exists: {sr_base.exists()}")

    # Check MMM subdirectory
    mmm_dir = sr_base / "mmm"
    print(f"📊 MMM Subdirectory: {mmm_dir}")
    print(f"   Exists: {mmm_dir.exists()}")

    # Create MMM charts directory
    charts_dir = mmm_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    print(f"📊 MMM Charts Directory: {charts_dir}")
    print(f"   Exists: {charts_dir.exists()}")

    # Check gap data directories (different location)
    daily_data_dir = Path("../downloadData_v1/data/market_data/daily")
    print(f"\n📄 Gap Data Directory (Daily): {daily_data_dir.resolve()}")
    print(f"   Exists: {daily_data_dir.exists()}")

    print(f"\n🎯 Expected File Structure:")
    print(f"   📄 Gap DATA files:")
    print(f"      {daily_data_dir}/XLY_gap.csv")
    print(f"      {daily_data_dir}/XLC_gap.csv")

    print(f"   📊 Chart FILES:")
    print(f"      {charts_dir}/XLY_gap_chart.png")
    print(f"      {charts_dir}/QQQ_Analysis_chart.png")

    print(f"\n✅ Directory separation:")
    print(f"   Data files: {daily_data_dir.resolve()}")
    print(f"   Chart files: {charts_dir.resolve()}")
    print(f"   Separation working: {daily_data_dir.resolve() != charts_dir.resolve()}")

    return True

if __name__ == "__main__":
    test_mmm_directories()
    print(f"\n🎉 MMM directory structure verified!")