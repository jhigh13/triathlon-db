import pandas as pd
import numpy as np

# 1. Load the cleaned file:
df = pd.read_csv("race_level_individual.csv")

# 2. Keep only the final overall rank & tier:
#    (Assumes you have a column called "Overall Rank" or similar.)
df = df[["Athlete", "Event", "Tier", "Finish Place"]]

# 3. Pivot to have one row per event, column per athlete:
pivot = df.pivot_table(
    index=["Event", "Tier"],
    columns="Athlete",
    values="Finish Place",
).dropna()   # only events where all athletes started

athletes = pivot.columns
n = len(athletes)
print(n)

# 4. Initialize win-count matrix:
wins = pd.DataFrame(0.0, index=athletes, columns=athletes)
counts = pd.DataFrame(0.0, index=athletes, columns=athletes)

# 5. Loop over events:
for (event, tier), row in pivot.iterrows():
    for i in athletes:
        for j in athletes:
            if i == j: continue
            counts.at[i,j] += tier
            if row[i] < row[j]:   # smaller rank = better
                wins.at[i,j] += tier

# 6. Compute % wins:
win_pct = wins / counts
win_pct.to_csv("head2head_win_pct.csv")
