import pandas as pd
import os

# File path
file_path = "USAT_MixedRelay_vJH.xlsx"

# Read all sheet names
all_sheets = pd.ExcelFile(file_path).sheet_names
all_sheets.remove("Sheet1")

# Define shortlist and tier mapping
short_list = [
    "Chase McQueen", "Morgan Pearson", "John Reed", 
    "Reese Vannerson", "Sullivan Middaugh",
    "Taylor Spivey", "Gwen Jorgensen", "Erika Ackerlund", "Keller Norland"
]

def ordinal_to_int(val):
    """Convert ordinal string like '2nd' to integer 2. 'Lap' or similar returns 0. Handles NaN and already-int values."""
    import re
    if pd.isna(val):
        return 0
    if isinstance(val, int):
        return val
    s = str(val).strip().lower()
    if s == "lap":
        return 0
    match = re.match(r"(\d+)", s)
    if match:
        return int(match.group(1))
    return 0

def load_and_clean(df: pd.DataFrame, sheet_name: str, tier: float) -> pd.DataFrame:
    # Normalize column names
    df.columns = df.columns.str.replace('\n', ' ').str.strip()
    # Identify & rename the athlete name column
    name_cols = [c for c in df.columns if "name" in c.lower()]
    if not name_cols:
        raise ValueError(f"No athlete column found in '{sheet_name}'")
    df = df.rename(columns={name_cols[0]: "Athlete"})
    # Add metadata
    df["Event"] = sheet_name
    df["Tier"] = tier
    # Convert ordinal place columns to integer if present
    place_cols = ["Swim Place", "T1 Place", "T2 Place", "Run Place", "Finish Place"]
    for col in place_cols:
        if col in df.columns:
            df[col] = df[col].apply(ordinal_to_int).astype(int)
    # Filter to shortlist
    df = df[df["Athlete"].isin(short_list)]
    return df

# Lists to collect cleaned data
cleaned_individual = []
cleaned_relaylegs = []

for sheet_name in all_sheets:
    # Determine tier weight
    tier = 1.0 if "WTCS" in sheet_name else 0.6

    if sheet_name == "WTCS Abu Dhabi":
        # Split into two tables by header rows (0 and 7)
        # Table 1: Individual Race (rows 0-6)
        df1 = pd.read_excel(
            file_path,
            sheet_name=sheet_name,
            header=1,
            nrows=6
        )
        # Table 2: Mixed Relay "similar leg" (rows 7+)
        df2 = pd.read_excel(
            file_path,
            sheet_name=sheet_name,
            header=8
        )
        # Clean both
        cleaned_individual.append(load_and_clean(df1, sheet_name, tier))
        cleaned_relaylegs.append(load_and_clean(df2, sheet_name + "_RelayLeg", tier))
    else:
        # Regular single table sheets
        df = pd.read_excel(
            file_path,
            sheet_name=sheet_name,
            header=0
        )
        cleaned_individual.append(load_and_clean(df, sheet_name, tier))

# Concatenate and drop all-null columns
race_ind_df = pd.concat(cleaned_individual, ignore_index=True).dropna(axis=1, how='all')
race_relay_df = pd.concat(cleaned_relaylegs, ignore_index=True).dropna(axis=1, how='all')

# Export to CSV
race_ind_df.to_csv("race_level_individual.csv", index=False)
race_relay_df.to_csv("race_level_relaylegs.csv", index=False)

print("Exported:")
print(" - race_level_individual.csv (individual race tables)")
print(" - race_level_relaylegs.csv (mixed relay leg tables)")
