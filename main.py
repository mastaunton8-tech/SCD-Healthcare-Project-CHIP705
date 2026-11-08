import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr

# Set up folders
Path("figures").mkdir(exist_ok=True)
Path("data").mkdir(exist_ok=True)

# Load CDC dataset
df = pd.read_csv("data/cdc_healthy_aging.csv")

# Select the 50 states and Washington, DC
states = [
    "AL",
    "AK",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "DE",
    "FL",
    "GA",
    "HI",
    "ID",
    "IL",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "ME",
    "MD",
    "MA",
    "MI",
    "MN",
    "MS",
    "MO",
    "MT",
    "NE",
    "NV",
    "NH",
    "NJ",
    "NM",
    "NY",
    "NC",
    "ND",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VT",
    "VA",
    "WA",
    "WV",
    "WI",
    "WY",
    "DC",
]

# Filter the two research indicators
data = df[
    (df["QuestionID"].isin(["Q30", "Q42"]))
    & (df["YearStart"].between(2015, 2022))
    & (df["Stratification1"] == "Overall")
    & (df["Stratification2"].isna())
    & (df["LocationAbbr"].isin(states))
    & (df["Data_Value_Type"] == "Percentage")
].copy()

# Keep relevant columns
data = data[["YearStart", "LocationAbbr", "LocationDesc", "QuestionID", "Data_Value"]]

# Organize the two indicators into separate columns
analysis = data.pivot_table(
    index=["YearStart", "LocationAbbr", "LocationDesc"],
    columns="QuestionID",
    values="Data_Value",
).reset_index()

analysis = analysis.rename(
    columns={
        "YearStart": "Year",
        "LocationAbbr": "State",
        "LocationDesc": "State_Name",
        "Q30": "SCD_Worsening",
        "Q42": "Healthcare_Discussion",
    }
)

# Save cleaned data
analysis.to_csv("data/analysis_dataset.csv", index=False)

# Print available data by year
print("\nNumber of states with both indicators by year:")
print(
    analysis.dropna(subset=["SCD_Worsening", "Healthcare_Discussion"])
    .groupby("Year")["State"]
    .nunique()
)

# Graph 1: Changes over time
yearly = analysis.groupby("Year")[["SCD_Worsening", "Healthcare_Discussion"]].mean()

yearly.plot(marker="o")
plt.title("SCD and Healthcare Engagement (2015–2022)")
plt.xlabel("Year")
plt.ylabel("Mean Percentage Across Reporting States")
plt.legend(["Worsening SCD", "Healthcare Discussion"])
plt.tight_layout()
plt.savefig("figures/yearly_trends.png")
plt.close()

# Find the year with the most complete state comparisons
complete = analysis.dropna(subset=["SCD_Worsening", "Healthcare_Discussion"])

best_year = complete.groupby("Year").size().idxmax()
comparison = complete[complete["Year"] == best_year].copy()

print("\nYear with the most complete state comparisons:", best_year)
print("Number of states:", len(comparison))

# Graph 2: Compare SCD across states
comparison = comparison.sort_values("SCD_Worsening")

plt.figure(figsize=(10, 12))
plt.barh(comparison["State_Name"], comparison["SCD_Worsening"])
plt.title(f"Worsening SCD by State ({best_year})")
plt.xlabel("Percentage")
plt.ylabel("State")
plt.tight_layout()
plt.savefig("figures/state_comparison.png")
plt.close()

# Graph 3: Relationship between SCD and healthcare discussion
plt.figure(figsize=(8, 6))
plt.scatter(comparison["SCD_Worsening"], comparison["Healthcare_Discussion"])
plt.title(f"SCD and Healthcare Discussion ({best_year})")
plt.xlabel("Worsening SCD (%)")
plt.ylabel("Healthcare Discussion (%)")
plt.tight_layout()
plt.savefig("figures/scd_healthcare_relationship.png")
plt.close()

# Calculate correlation
correlation, p_value = pearsonr(
    comparison["SCD_Worsening"], comparison["Healthcare_Discussion"]
)

print("\nCorrelation Results")
print("Pearson correlation:", round(correlation, 3))
print("P-value:", round(p_value, 4))

print("\nAnalysis complete. Graphs saved in the figures folder.")
