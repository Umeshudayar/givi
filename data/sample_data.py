#!/usr/bin/env python3
"""
Generate sample delivery data for training the LSTM model
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.helpers import generate_sample_data

def main():
    # Generate sample data
    df = generate_sample_data(5000)
    
    # Save to CSV
    output_path = "data/processed/delivery_data.csv"
    df.to_csv(output_path, index=False)
    print(f"Sample data saved to {output_path}")
    
    # Print statistics
    print("\nData Statistics:")
    print(f"Total records: {len(df)}")
    print(f"Average delivery time: {df['delivery_time'].mean():.2f} minutes")
    print(f"Delivery time range: {df['delivery_time'].min():.1f} - {df['delivery_time'].max():.1f} minutes")

if __name__ == "__main__":
    main()
