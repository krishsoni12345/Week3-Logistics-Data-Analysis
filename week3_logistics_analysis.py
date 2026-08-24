# Week 3 - Advanced Data Analysis and Visualization in Logistics
# The script below creates a hypothetical logistics dataset, performs EDA,
# and generates visualizations.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

n = 250
regions = np.random.choice(["North", "South", "East", "West"], n)
modes = np.random.choice(["Road", "Rail", "Air"], n, p=[0.60, 0.25, 0.15])
priorities = np.random.choice(["Standard", "Express"], n, p=[0.75, 0.25])

shipment_volume = np.random.randint(20, 501, n)
distance_km = np.random.randint(50, 2001, n)

base_days = distance_km / 500
mode_add = np.where(modes == "Road", 1.2, np.where(modes == "Rail", 2.0, 0.3))
priority_adj = np.where(priorities == "Express", -0.8, 0)

delivery_time = np.maximum(
    1,
    np.round(base_days + mode_add + priority_adj +
             np.random.normal(0, 0.8, n), 1)
)

transport_rate = np.where(
    modes == "Road", 5.2,
    np.where(modes == "Rail", 3.7, 11.5)
)

transport_cost = (
    distance_km * shipment_volume / 100
    * transport_rate
    * np.random.uniform(0.88, 1.12, n)
)
transport_cost = np.round(transport_cost, 2)

delay_probability = np.clip(
    0.08 + (distance_km / 2000) * 0.10 +
    (modes == "Road") * 0.07 +
    (shipment_volume > 400) * 0.05,
    0.03, 0.45
)

delayed = np.random.binomial(1, delay_probability)
delay_status = np.where(delayed == 1, "Delayed", "On Time")

customer_rating = np.clip(
    4.6 - delivery_time * 0.12 -
    delayed * 0.35 +
    np.random.normal(0, 0.25, n),
    1, 5
)
customer_rating = np.round(customer_rating, 1)

data = pd.DataFrame({
    "Shipment_ID": [f"SHP{1001+i}" for i in range(n)],
    "Region": regions,
    "Transport_Mode": modes,
    "Priority": priorities,
    "Shipment_Volume_kg": shipment_volume,
    "Distance_km": distance_km,
    "Delivery_Time_days": delivery_time,
    "Transport_Cost_INR": transport_cost,
    "Delay_Status": delay_status,
    "Customer_Rating": customer_rating
})

# Basic EDA
print(data.head())
print(data.info())
print(data.describe())

print("\nMissing values:")
print(data.isnull().sum())

print("\nAverage delivery time:", data["Delivery_Time_days"].mean())
print("Median delivery time:", data["Delivery_Time_days"].median())
print("Average transport cost:", data["Transport_Cost_INR"].mean())

print("\nCorrelation matrix:")
print(data.select_dtypes(include=np.number).corr())

# Visualization 1
plt.hist(data["Delivery_Time_days"], bins=18, edgecolor="black")
plt.title("Distribution of Delivery Times")
plt.xlabel("Delivery Time (days)")
plt.ylabel("Number of Shipments")
plt.show()

# Visualization 2
mode_avg = data.groupby("Transport_Mode")["Delivery_Time_days"].mean()
plt.bar(mode_avg.index, mode_avg.values)
plt.title("Average Delivery Time by Transport Mode")
plt.xlabel("Transport Mode")
plt.ylabel("Average Delivery Time (days)")
plt.show()

# Visualization 3
for mode in ["Road", "Rail", "Air"]:
    subset = data[data["Transport_Mode"] == mode]
    plt.scatter(
        subset["Distance_km"],
        subset["Transport_Cost_INR"],
        alpha=0.55,
        label=mode
    )

plt.title("Transportation Cost vs Distance")
plt.xlabel("Distance (km)")
plt.ylabel("Transport Cost (INR)")
plt.legend()
plt.show()

# Visualization 4
region_delay = data.groupby("Region")["Delay_Status"].apply(
    lambda x: (x == "Delayed").mean() * 100
)
plt.bar(region_delay.index, region_delay.values)
plt.title("Delay Rate by Region")
plt.xlabel("Region")
plt.ylabel("Delayed Shipments (%)")
plt.show()

# Visualization 5
corr = data.select_dtypes(include=np.number).corr()
plt.imshow(corr, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
plt.colorbar(label="Correlation")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Correlation Matrix")
plt.show()
