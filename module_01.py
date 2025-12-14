import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_excel("Strava Running Data.xlsx")

df = df[df["sport_type"].str.contains("Run", case=False, na=False)].copy()

df["distance_km"] = df["distance"] / 1000
df["moving_time_min"] = df["moving_time"] / 60
df["pace_min_per_km"] = df["moving_time_min"] / df["distance_km"]

df = df[df["pace_min_per_km"].between(3, 15)]
df = df[df["distance_km"].between(0.5, 60)]

def classify(row):
    if row["distance_km"] >= 12:
        return "Run-Long"
    elif row["distance_km"] <= 5 and row["pace_min_per_km"] > df["pace_min_per_km"].median():
        return "Run-Recovery"
    elif row["pace_min_per_km"] < 5:
        return "Run-Intervals"
    else:
        return "Run-Easy"

df["workout_type"] = df.apply(classify, axis=1)
df["index"] = 1
df["year"] = pd.to_datetime(df["start_date"]).dt.year

sns.set_theme(style="whitegrid")

# Runs by Workout Type
plt.figure(figsize=(12,6))
sns.lineplot(data=df, x="year", y="index", hue="workout_type",
             estimator="sum", ci=None)
plt.title("Total Runs by Workout Type Over Time")
plt.ylabel("Runs")
plt.show()

# Total Distance
plt.figure(figsize=(12,6))
sns.lineplot(data=df, x="year", y="distance_km", hue="workout_type",
             estimator="sum", ci=None)
plt.title("Total Distance (km) by Workout Type Over Time")
plt.ylabel("Distance (km)")
plt.show()

# Average Pace
plt.figure(figsize=(12,6))
sns.lineplot(data=df, x="year", y="pace_min_per_km", hue="workout_type",
             estimator="mean", ci=None)
plt.title("Average Training Pace by Workout Type")
plt.ylabel("Pace (min/km)")
plt.show()

# Pace Distribution
plt.figure(figsize=(12,6))
sns.boxplot(data=df, x="workout_type", y="pace_min_per_km")
plt.title("Pace Distribution by Workout Type")
plt.ylabel("Pace (min/km)")
plt.show()