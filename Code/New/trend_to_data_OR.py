from pytrends.request import TrendReq
import pandas as pd
import time

# Read terms from file
with open('ShiHaoYang/Data/US_terms.txt') as f:
    lines = [line.strip() for line in f if line.strip()]

# Build a set of all unique terms (split on '+')
unique_terms = set()
groups = []
for line in lines:
    terms = [t.strip() for t in line.split('+')]
    groups.append(terms)
    unique_terms.update(terms)

unique_terms = list(unique_terms)
print(f"Total unique terms: {len(unique_terms)}")

# Batch unique terms into groups of 5
batches = [unique_terms[i:i+5] for i in range(0, len(unique_terms), 5)]

pytrends = TrendReq(hl='en-US', tz=360)
all_data = pd.DataFrame()

for i, batch in enumerate(batches):
    try:
        pytrends.build_payload(batch, timeframe='2022-01-01 2024-12-31', geo='US')
        df = pytrends.interest_over_time().drop(columns=['isPartial'])
        if all_data.empty:
            all_data = df[batch]
        else:
            all_data = all_data.join(df[batch], how='outer')
        print(f"Batch {i+1}/{len(batches)} completed: {batch}")
        time.sleep(1)
    except Exception as e:
        print(f"Batch {i+1}/{len(batches)} failed: {batch} ({e})")

# Now, create group columns by summing the relevant terms
for group in groups:
    group_name = ' + '.join(group)
    missing = [t for t in group if t not in all_data.columns]
    if missing:
        print(f"Warning: Missing terms for group {group_name}: {missing}")
        continue
    all_data[group_name] = all_data[group].sum(axis=1)

# Save only the group columns (or keep all if you want)
group_names = [' + '.join(g) for g in groups]
all_data[group_names].to_csv("ShiHaoYang/Data/trends_us_data_grouped.csv")
print("Saved to trends_us_data_grouped.csv")