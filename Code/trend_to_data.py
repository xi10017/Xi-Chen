from pytrends.request import TrendReq
import pandas as pd
import time

# Replace this with your cleaned list
terms_cleaned = [
    "influenza type a", "get over the flu", "type a influenza", "flu care",
    "symptoms of flu", "treating flu", "i have the flu", "how long contagious",
    "flu duration", "flu vs cold", "taking temperature", "fight the flu",
    "flu contagious", "having the flu", "flu versus cold", "reduce a fever",
    "flu fever", "treatment for flu", "bronchitis", "cure the flu",
    "treat the flu", "human temperature", "how long flu", "medicine for flu",
    "how to treat the flu", "dangerous fever", "flu germs", "flu length",
    "signs of the flu", "the flu", "cold vs flu", "cure flu",
    "over the counter flu", "remedies for flu", "flu and cold", "exposed to flu",
    "how long is the flu", "influenza a and b", "thermoscan", "low body",
    "symptoms of the flu", "contagious flu", "flu complications", "early flu symptoms",
    "flu recovery", "how long does the flu last", "high fever", "remedies for the flu",
    "cold or flu", "fever flu", "flu children", "flu report",
    "flu medicine", "oscillococcinum", "the flu virus", "incubation period for flu",
    "flu or cold", "flu remedies", "how to treat flu", "break a fever",
    "normal body", "how long is flu contagious", "pneumonia", "flu contagious period",
    "is flu contagious", "flu treatments", "flu headache", "influenza incubation period",
    "treat flu", "influenza symptoms", "flu cough", "cold versus flu",
    "body temperature", "cold vs flu", "ear thermometer", "flu in children",
    "is the flu contagious", "braun thermoscan", "how to get rid of the flu", "what to do if you have the flu",
    "reduce fever", "fever cough", "flu how long", "medicine for the flu",
    "flu treatment", "signs of flu", "symptoms of bronchitis", "flu and fever",
    "flu vs cold", "how long does flu last", "cold and flu", "flu lasts",
    "how long is the flu contagious", "normal body temperature", "over the counter flu medicine", "incubation period for the flu",
    "fever reducer", "get rid of the flu", "treating the flu", "do i have the flu"
]

# Group terms into batches of 5
term_batches = [terms_cleaned[i:i + 5] for i in range(0, len(terms_cleaned), 5)]

# Initialize pytrends
pytrends = TrendReq(hl='en-US', tz=360)
all_data = pd.DataFrame()

# Query each batch
for i, batch in enumerate(term_batches):
    try:
        pytrends.build_payload(batch, timeframe='2022-01-01 2024-12-31', geo='CA')
        df = pytrends.interest_over_time().drop(columns=['isPartial'])

        if all_data.empty:
            all_data = df
        else:
            all_data = all_data.join(df, how='outer')

        print(f"Batch {i+1} completed: {batch}")
        time.sleep(2)
    except Exception as e:
        print(f"Batch {i+1} failed: {e}")

# Save to CSV
all_data.to_csv("ShiHaoYang/all_google_trends_data.csv")
print("Saved to all_google_trends_data.csv")