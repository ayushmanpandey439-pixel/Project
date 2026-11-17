import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. LOAD & INSPECT RAW DATA
df = pd.read_csv("raw_news_data.csv")
initial_rows = len(df)  # Save original row count for final comparison

print("Dataset Shape:", df.shape)
print("\nData Types & Non-Null Counts:")
df.info()
print("\nMissing Values:\n", df.isnull().sum())
print("\nExact Duplicates:", df.duplicated().sum())
print("\nSummary Statistics:\n", df.describe())
# 2. DATA CLEANING
# Step 1: Remove exact duplicate rows
df = df.drop_duplicates()

# Step 2: Standardize text formatting
text_cols = ["label", "channel_type", "category", "region"]
for col in text_cols:
    df[col] = df[col].str.strip().str.lower()

df["channel_type"] = df["channel_type"].str.replace(" ", "_")
df["headline"] = df["headline"].str.strip()
df["channel_name"] = df["channel_name"].str.strip()

# Step 3: Make channel_name and channel_type consistent per channel_id
# Helper function to safely get the most frequent value (mode)
def get_most_common(series):
    valid_values = series.dropna()
    return valid_values.mode()[0] if len(valid_values) > 0 else np.nan

df["channel_name"] = df.groupby("channel_id")["channel_name"].transform(get_most_common)
df["channel_type"] = df.groupby("channel_id")["channel_type"].transform(get_most_common)
print("\nUnique Channels (should be 150):", df["channel_name"].nunique())

# Step 4: Convert follower strings (e.g., "12.5K") into numbers
def convert_followers(val):
    if pd.isnull(val):
        return np.nan
    val = str(val).strip().upper()
    if val.endswith("K"):
        return float(val[:-1]) * 1000
    return float(val)

df["channel_followers"] = df["channel_followers"].apply(convert_followers)

# Step 5: Standardize post_time across 3 date formats & drop future dates
t1 = pd.to_datetime(df["post_time"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
t2 = pd.to_datetime(df["post_time"], format="%d/%m/%Y %H:%M", errors="coerce")
t3 = pd.to_datetime(df["post_time"], format="%d-%b-%Y %I:%M %p", errors="coerce")

df["post_time"] = t1.fillna(t2).fillna(t3)

# Keep only dates up to mid-October 2026
df = df[df["post_time"] <= "2026-10-17"]

# Step 6: Fix negative or impossible numeric values
df["likes"] = df["likes"].abs()
df.loc[df["account_age_days"] < 0, "account_age_days"] = np.nan
df.loc[df["views"] < df["shares"], "views"] = np.nan

# Step 7: Create a standardized headline key for matching duplicates/categories
df["headline_key"] = df["headline"].str.lower().str.strip().str.rstrip(".")

# Step 8: Fill missing numeric values with channel median, then overall median
num_cols = ["channel_followers", "account_age_days", "views", "likes", "comments"]
for col in num_cols:
    channel_median = df.groupby("channel_id")[col].transform("median")
    df[col] = df[col].fillna(channel_median).fillna(df[col].median())

# Step 9: Fill missing categories (by matching headline) and missing regions
df["category"] = df.groupby("headline_key")["category"].transform(get_most_common)
df["category"] = df["category"].fillna("unknown")
df["region"] = df["region"].fillna("unknown")

# Step 10: Fill missing labels from parent_post_id, then drop remaining nulls
label_lookup = df.dropna(subset=["label"]).drop_duplicates("post_id").set_index("post_id")["label"]
df["label"] = df["label"].fillna(df["parent_post_id"].map(label_lookup))
df = df.dropna(subset=["label"])

# Step 11: Drop unused 'content' column and near-duplicate posts
df = df.drop(columns=["content"])
df = df.drop_duplicates(subset=["channel_id", "headline_key", "post_time"])


# ==========================================
# 3. FEATURE ENGINEERING
# ==========================================
df["is_fake"] = df["label"] == "fake"
df["post_date"] = df["post_time"].dt.date
df["hour"] = df["post_time"].dt.hour
df["weekday"] = df["post_time"].dt.day_name()

# Avoid division by zero if views == 0
df["engagement_rate"] = (df["likes"] + df["shares"] + df["comments"]) / df["views"].replace(0, np.nan)

# Final Verification & Save
print("\n--- CLEANING COMPLETE ---")
print("Remaining Nulls:\n", df.isnull().sum())
print("Duplicates Left:", df.duplicated().sum())
print(f"Rows: {initial_rows} -> {len(df)}")

df.to_csv("clean_news_data.csv", index=False)


# ==========================================
# 4. STATISTICAL SUMMARY TABLES
# ==========================================
print("\n--- DATA ANALYSIS SUMMARY ---")
print("\n1. Post Counts by Label:\n", df["label"].value_counts())

print("\n2. Average Metrics (Fake vs. Genuine):")
print(df.groupby("label")[["views", "likes", "shares", "comments", "engagement_rate", "channel_followers", "account_age_days"]].mean().round(2))

print("\n3. Fake News Percentage by Category (%):")
print((df.groupby("category")["is_fake"].mean() * 100).round(1).sort_values(ascending=False))

print("\n4. Fake News Percentage by Channel Type (%):")
print((df.groupby("channel_type")["is_fake"].mean() * 100).round(1).sort_values(ascending=False))


# 5. COMPLETE VISUAL DATA ANALYSIS (8 GRAPHS
plt.style.use("ggplot")  # Clean visual style for all charts

# --- Graph 1: Overall Distribution of Fake vs Genuine News ---
plt.figure(figsize=(7, 5))
label_counts = df["label"].value_counts()
bars = plt.bar(label_counts.index, label_counts.values, color=["steelblue", "tomato"])
plt.title("1. Distribution of Fake and Genuine News")
plt.xlabel("News Label")
plt.ylabel("Number of Posts")
for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{int(bar.get_height())}", ha="center", va="bottom")
plt.tight_layout()
plt.show()

# --- Graph 2: Fake vs Genuine News by Category ---
category_ct = pd.crosstab(df["category"], df["label"])
category_ct.plot(kind="bar", figsize=(9, 5), color=["tomato", "steelblue"])
plt.title("2. Fake vs. Genuine Posts by News Category")
plt.xlabel("Category")
plt.ylabel("Number of Posts")
plt.xticks(rotation=30)
plt.legend(title="Label")
plt.tight_layout()
plt.show()

# --- Graph 3: Fake vs Genuine News by Channel Type ---
channel_ct = pd.crosstab(df["channel_type"], df["label"])
channel_ct.plot(kind="bar", figsize=(9, 5), color=["tomato", "steelblue"])
plt.title("3. Fake vs. Genuine Posts by Channel Type")
plt.xlabel("Channel Type")
plt.ylabel("Number of Posts")
plt.xticks(rotation=30)
plt.legend(title="Label")
plt.tight_layout()
plt.show()

# --- Graph 4: Fake News Rate (%) by Region ---
plt.figure(figsize=(9, 5))
region_fake_pct = (df.groupby("region")["is_fake"].mean() * 100).sort_values(ascending=False)
region_fake_pct.plot(kind="bar", color="coral")
plt.title("4. Percentage of Fake News by Region")
plt.xlabel("Region")
plt.ylabel("Fake News Percentage (%)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# --- Graph 5: Posting Activity by Hour of Day ---
hourly_ct = pd.crosstab(df["hour"], df["label"])
hourly_ct.plot(kind="line", marker="o", figsize=(10, 5), color=["tomato", "steelblue"])
plt.title("5. Hourly Posting Trends: Fake vs. Genuine News")
plt.xlabel("Hour of Day (0 - 23)")
plt.ylabel("Number of Posts")
plt.xticks(range(0, 24))
plt.grid(True)
plt.legend(title="Label")
plt.tight_layout()
plt.show()

# --- Graph 6: Posting Activity by Day of the Week ---
day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
weekday_ct = pd.crosstab(df["weekday"], df["label"]).reindex(day_order)
weekday_ct.plot(kind="bar", figsize=(9, 5), color=["tomato", "steelblue"])
plt.title("6. Posting Volume by Day of the Week")
plt.xlabel("Day of the Week")
plt.ylabel("Number of Posts")
plt.xticks(rotation=30)
plt.legend(title="Label")
plt.tight_layout()
plt.show()

# --- Graph 7: Distribution of Engagement Rate ---
plt.figure(figsize=(8, 5))
plt.hist(df[df["label"] == "genuine"]["engagement_rate"].dropna(), bins=40, alpha=0.6, label="Genuine", color="steelblue")
plt.hist(df[df["label"] == "fake"]["engagement_rate"].dropna(), bins=40, alpha=0.6, label="Fake", color="tomato")
plt.title("7. Distribution of Engagement Rate (Fake vs. Genuine)")
plt.xlabel("Engagement Rate")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.show()

# --- Graph 8: Average Likes, Shares & Comments Comparison ---
avg_interactions = df.groupby("label")[["likes", "shares", "comments"]].mean()
avg_interactions.plot(kind="bar", figsize=(8, 5), color=["#4c72b0", "#dd8452", "#55a868"])
plt.title("8. Average User Interactions per Post")
plt.xlabel("News Label")
plt.ylabel("Average Count")
plt.xticks(rotation=0)
plt.legend(title="Interaction Type")
plt.tight_layout()
plt.show()