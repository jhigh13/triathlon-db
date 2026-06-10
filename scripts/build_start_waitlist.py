"""
Generate simulated start list and waitlist from registered athletes.

Rules:
- Start list = registered athletes ordered by world ranking (2026-06-01)
- Max 5 athletes per country on the start list
- Athletes bumped by the country cap go to waitlist (ordered by world ranking)
- Athletes with no world ranking go to bottom of waitlist
"""
import pandas as pd
from tri_analysis.database import get_engine

EXCEL_PATH = "data/waitlist_195148.xlsx"
RANKING_DATE = "2026-06-01"
COUNTRY_CAP = 5
BRAXTON_LEGG = {
    "Member ID": None,  # unknown, will search
    "First Name": "Braxton",
    "Last Name": "Legg",
    "Country": "USA",
    "Gender": "male",
    "Program ID": 678086,
    "Program Name": "Elite Men",
}

engine = get_engine()

# --- Load Excel ---
df = pd.read_excel(EXCEL_PATH)
df = df.rename(columns={"Member ID": "athlete_id"})

# Focus on Elite Men and Elite Women only
elite = df[df["Program Name"].isin(["Elite Men", "Elite Women"])].copy()
print(f"Loaded {len(elite)} elite athletes from Excel")

# --- Check / add Braxton Legg ---
braxton_in_list = (
    elite["First Name"].str.lower().str.strip() == "braxton"
) & (
    elite["Last Name"].str.lower().str.strip() == "legg"
)
if braxton_in_list.any():
    print("Braxton Legg is already in the list.")
else:
    print("Braxton Legg NOT found in list — searching database for his athlete_id...")
    with engine.connect() as conn:
        result = pd.read_sql(
            "SELECT athlete_id, full_name, country FROM athlete WHERE LOWER(full_name) LIKE '%%legg%%' AND LOWER(full_name) LIKE '%%braxton%%'",
            conn
        )
    if result.empty:
        with engine.connect() as conn:
            result = pd.read_sql(
                "SELECT athlete_id, full_name, country FROM athlete WHERE LOWER(full_name) LIKE '%%braxton%%'",
                conn
            )
    print("  DB search results:", result.to_string() if not result.empty else "Not found in athlete table")
    if not result.empty:
        bid = int(result.iloc[0]["athlete_id"])
        bname = result.iloc[0]["full_name"].strip()
        parts = bname.split()
        BRAXTON_LEGG["Member ID"] = bid
        BRAXTON_LEGG["athlete_id"] = bid
        print(f"  Found: {bname} (athlete_id={bid})")
    else:
        BRAXTON_LEGG["athlete_id"] = None
        print("  Not found in DB — adding with unknown ID.")
    
    row = pd.DataFrame([{
        "athlete_id": BRAXTON_LEGG.get("athlete_id"),
        "First Name": BRAXTON_LEGG["First Name"],
        "Last Name": BRAXTON_LEGG["Last Name"],
        "Country": BRAXTON_LEGG["Country"],
        "Gender": BRAXTON_LEGG["Gender"],
        "Program ID": BRAXTON_LEGG["Program ID"],
        "Program Name": BRAXTON_LEGG["Program Name"],
        "Waitlist Position": None,
        "Notes": "Added manually",
    }])
    elite = pd.concat([elite, row], ignore_index=True)
    print(f"  Added Braxton Legg to the list.")

# --- Fetch world rankings from DB ---
with engine.connect() as conn:
    # Men: ranking_cat_id=13, Women: ranking_cat_id=14
    rankings = pd.read_sql(
        f"""
        SELECT athlete_id, athlete_name, ranking_cat_id, rank_position, total_points
        FROM athlete_rankings
        WHERE retrieved_at = '{RANKING_DATE}'
          AND ranking_cat_id IN (13, 14)
        """,
        conn
    )
print(f"Fetched {len(rankings)} ranking records for {RANKING_DATE}")

# Map gender to ranking category
cat_map = {"male": 13, "female": 14}
elite["ranking_cat_id"] = elite["Gender"].str.lower().str.strip().map(cat_map)

# Merge rankings onto athletes
elite_with_rank = elite.merge(
    rankings[["athlete_id", "ranking_cat_id", "rank_position", "total_points"]],
    on=["athlete_id", "ranking_cat_id"],
    how="left"
)

# Identify athletes not found in ranking table
not_ranked = elite_with_rank[elite_with_rank["rank_position"].isna()][["athlete_id", "First Name", "Last Name", "Country", "Program Name"]]
if not not_ranked.empty:
    print(f"\nAthletes with no world ranking ({len(not_ranked)}):")
    print(not_ranked.to_string(index=False))

# --- Build start lists and waitlists ---
def build_start_waitlist(group_df, gender_label, country_cap=5, start_limit=55):
    # Sort: ranked athletes first (by rank_position asc), unranked last
    ranked = group_df[group_df["rank_position"].notna()].sort_values("rank_position")
    unranked = group_df[group_df["rank_position"].isna()].sort_values(["Country", "Last Name"])

    start_list = []
    wait_list = []
    country_counts = {}

    # Process ranked athletes in order
    for _, row in ranked.iterrows():
        country = row["Country"]
        count = country_counts.get(country, 0)
        if len(start_list) < start_limit and count < country_cap:
            country_counts[country] = count + 1
            start_list.append(row)
        else:
            wait_list.append(row)

    # All unranked go to waitlist at the bottom
    for _, row in unranked.iterrows():
        wait_list.append(row)

    # Build output dataframes
    cols = ["rank_position", "total_points", "athlete_id", "First Name", "Last Name", "Country", "Gender"]

    sl_df = pd.DataFrame(start_list)[cols].reset_index(drop=True)
    sl_df.insert(0, "Start Position", range(1, len(sl_df) + 1))

    wl_df = pd.DataFrame(wait_list)[cols].reset_index(drop=True) if wait_list else pd.DataFrame(columns=["Waitlist Position"] + cols)
    if len(wl_df) > 0:
        wl_df.insert(0, "Waitlist Position", range(1, len(wl_df) + 1))

    return sl_df, wl_df

# Split by gender
men_df = elite_with_rank[elite_with_rank["Program Name"] == "Elite Men"].copy()
women_df = elite_with_rank[elite_with_rank["Program Name"] == "Elite Women"].copy()

men_start, men_wait = build_start_waitlist(men_df, "Male")
women_start, women_wait = build_start_waitlist(women_df, "Female")

# --- Display results ---
print("\n" + "="*80)
print(f"ELITE MEN — START LIST ({len(men_start)} athletes)")
print("="*80)
print(men_start.rename(columns={
    "rank_position": "World Rank", "total_points": "Points",
    "athlete_id": "ID", "First Name": "First", "Last Name": "Last"
}).to_string(index=False))

print(f"\nELITE MEN — WAITLIST ({len(men_wait)} athletes)")
print("-"*80)
if len(men_wait) > 0:
    print(men_wait.rename(columns={
        "rank_position": "World Rank", "total_points": "Points",
        "athlete_id": "ID", "First Name": "First", "Last Name": "Last"
    }).to_string(index=False))
else:
    print("(empty)")

print("\n" + "="*80)
print(f"ELITE WOMEN — START LIST ({len(women_start)} athletes)")
print("="*80)
print(women_start.rename(columns={
    "rank_position": "World Rank", "total_points": "Points",
    "athlete_id": "ID", "First Name": "First", "Last Name": "Last"
}).to_string(index=False))

print(f"\nELITE WOMEN — WAITLIST ({len(women_wait)} athletes)")
print("-"*80)
if len(women_wait) > 0:
    print(women_wait.rename(columns={
        "rank_position": "World Rank", "total_points": "Points",
        "athlete_id": "ID", "First Name": "First", "Last Name": "Last"
    }).to_string(index=False))
else:
    print("(empty)")

# --- Save to Excel ---
out_path = "outputs/start_waitlist_analysis.xlsx"
with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
    men_start.to_excel(writer, sheet_name="Men Start List", index=False)
    men_wait.to_excel(writer, sheet_name="Men Waitlist", index=False)
    women_start.to_excel(writer, sheet_name="Women Start List", index=False)
    women_wait.to_excel(writer, sheet_name="Women Waitlist", index=False)

print(f"\nSaved to {out_path}")

# --- Country summary ---
print("\n=== COUNTRY COUNTS ON START LISTS ===")
men_country = men_start.groupby("Country").size().sort_values(ascending=False)
print("Men:")
print(men_country.to_string())
women_country = women_start.groupby("Country").size().sort_values(ascending=False)
print("\nWomen:")
print(women_country.to_string())
