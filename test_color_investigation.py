#!/usr/bin/env python3
"""
Small test to investigate the color assignment issue.
User reports seeing "red line" but our tests show "blue line".
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime

# Test with and without seaborn palette that SR system uses
print("🎨 COLOR INVESTIGATION TEST")
print("=" * 50)

# Create simple test data
dates = pd.date_range('2024-01-01', '2024-01-10', freq='D')
prices = [100 + i * 0.5 for i in range(len(dates))]
series = pd.Series(prices, index=dates, name='QQQ')

# Test 1: Default matplotlib colors
print("\n1. DEFAULT MATPLOTLIB COLORS:")
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
line = ax.plot(series.index, series.values, label='QQQ', linewidth=1.5, alpha=0.8)
ax.legend()
ax.set_title("Default Colors (No Seaborn)")
color = line[0].get_color()
print(f"   Line color: {color}")
plt.savefig('test_color_default.png', dpi=150, bbox_inches='tight')
plt.close()

# Test 2: With seaborn palette (like SR system)
print("\n2. WITH SEABORN 'husl' PALETTE (SR System Style):")
sns.set_palette("husl")  # This is what SR system uses
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
line = ax.plot(series.index, series.values, label='QQQ', linewidth=1.5, alpha=0.8)
ax.legend()
ax.set_title("With Seaborn 'husl' Palette")
color = line[0].get_color()
print(f"   Line color: {color}")
plt.savefig('test_color_seaborn.png', dpi=150, bbox_inches='tight')
plt.close()

# Test 3: Explicit blue color
print("\n3. EXPLICIT BLUE COLOR:")
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
line = ax.plot(series.index, series.values, label='QQQ', linewidth=1.5, alpha=0.8, color='blue')
ax.legend()
ax.set_title("Explicit Blue Color")
color = line[0].get_color()
print(f"   Line color: {color}")
plt.savefig('test_color_blue.png', dpi=150, bbox_inches='tight')
plt.close()

# Test 4: Check seaborn palette colors
print("\n4. SEABORN 'husl' PALETTE COLORS:")
sns.set_palette("husl")
palette = sns.color_palette()
for i, color in enumerate(palette[:6]):
    print(f"   Color {i}: {color}")

print("\n5. HYPOTHESIS:")
print("   - User's system uses seaborn 'husl' palette")
print("   - First color in 'husl' palette might appear reddish")
print("   - User sees 'red' due to color perception/display differences")
print("   - Our 'blue' specification should override palette")

print("\n6. CONCLUSION:")
print("   Check generated images:")
print("   - test_color_default.png (matplotlib default)")
print("   - test_color_seaborn.png (with husl palette)")
print("   - test_color_blue.png (explicit blue)")
print("   Colors may appear different due to:")
print("   - Display calibration")
print("   - Color perception")
print("   - Theme/dark mode settings")