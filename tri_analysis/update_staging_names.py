#!/usr/bin/env python3
"""
Update staging_rankings table with manual name mappings from CSV file.
"""
import sys
import os
# Add project root to path for package imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import csv
import pandas as pd
from sqlalchemy import text
from Data_Import.database import get_engine

def update_staging_rankings_from_csv(csv_file_path='Data_Import/names_mapping.csv'):
    """
    Update staging_rankings table using mappings from CSV file.
    
    Args:
        csv_file_path: Path to CSV file with columns: old_name, athlete_name, athlete_id
    """
    engine = get_engine()
    
    # Read the CSV file
    try:
        df = pd.read_csv(csv_file_path)
        # Clean up any whitespace in column names
        df.columns = df.columns.str.strip()
        print(f"Loaded {len(df)} name mappings from {csv_file_path}")
        print(df)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    
    updated_count = 0
    
    with engine.begin() as conn:
        for _, row in df.iterrows():
            old_name = str(row['old_name']).strip()
            athlete_name = str(row['athlete_name']).strip()
            
            # Skip rows with missing athlete_id
            if pd.isna(row['athlete_id']) or str(row['athlete_id']).strip() == '':
                print(f"⚠ Skipping '{old_name}' - missing athlete_id")
                continue
                
            try:
                athlete_id = int(row['athlete_id'])
            except (ValueError, TypeError):
                print(f"⚠ Skipping '{old_name}' - invalid athlete_id: {row['athlete_id']}")
                continue
            
            # Update query
            update_sql = text("""
                UPDATE staging_rankings 
                SET athlete_name = :new_name, 
                    athlete_id = :athlete_id
                WHERE athlete_name = :old_name 
                  AND athlete_id IS NULL
            """)
            
            result = conn.execute(update_sql, {
                'new_name': athlete_name,
                'athlete_id': athlete_id,
                'old_name': old_name
            })
            
            rows_affected = result.rowcount
            updated_count += rows_affected
            
            if rows_affected > 0:
                print(f"✓ Updated {rows_affected} records: '{old_name}' -> '{athlete_name}' (ID: {athlete_id})")
            else:
                print(f"⚠ No records found for '{old_name}' or already updated")
    
    print(f"\nTotal records updated: {updated_count}")
    
    # Show final results
    check_sql = text("""
        SELECT athlete_name, athlete_id, COUNT(*) as record_count
        FROM staging_rankings 
        WHERE athlete_name IN :names
        GROUP BY athlete_name, athlete_id
        ORDER BY athlete_name
    """)
    
    updated_names = tuple(df['athlete_name'].str.strip().tolist())
    with engine.connect() as conn:
        results = conn.execute(check_sql, {'names': updated_names}).fetchall()
        
    if results:
        print("\nFinal state of updated records:")
        for row in results:
            print(f"  {row.athlete_name}: ID {row.athlete_id} ({row.record_count} records)")

if __name__ == "__main__":
    update_staging_rankings_from_csv()
