import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# 1. Load the dataset
df = pd.read_csv('crib.csv')

# 2. Clean the data
# Drop missing values for the features we care about
df_clean = df.dropna(subset=['Price', 'Bedrooms', 'Bathrooms']).copy()

# Remove extreme outliers in price (top 5%) to make the clusters readable
q_high = df_clean['Price'].quantile(0.95)
df_clean = df_clean[df_clean['Price'] <= q_high]

# 3. Prepare features and Scale
features = ['Price', 'Bedrooms', 'Bathrooms']
X = df_clean[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Run K-Means Clustering (K=3)
kmeans = KMeans(n_clusters=3, random_state=42)
df_clean['Cluster'] = kmeans.fit_predict(X_scaled)

# Calculate Silhouette Score for validation
sil_score = silhouette_score(X_scaled, df_clean['Cluster'])
print(f"Silhouette Score: {sil_score:.3f}")

# Rename clusters for better readability
cluster_mapping = {
    df_clean.groupby('Cluster')['Price'].mean().idxmin(): 'Tier 1: Starter / Investor',
    df_clean.groupby('Cluster')['Price'].mean().idxmax(): 'Tier 3: Premium Luxury',
}
for c in [0, 1, 2]:
    if c not in cluster_mapping:
        cluster_mapping[c] = 'Tier 2: Mid-Market Family'

df_clean['Buyer Segment'] = df_clean['Cluster'].map(cluster_mapping)

# Print Summary Stats
print("\n--- Cluster Summary ---")
print(df_clean.groupby('Buyer Segment')[['Price', 'Bedrooms', 'Bathrooms']].mean().round(1))

# 5. Create Visualization
plt.figure(figsize=(10, 6))
sns.set_theme(style="whitegrid")

custom_palette = {'Tier 1: Starter / Investor': '#1f77b4', 
                  'Tier 2: Mid-Market Family': '#ff7f0e', 
                  'Tier 3: Premium Luxury': '#2ca02c'}

scatter = sns.scatterplot(data=df_clean, x='Bedrooms', y='Price', 
                          hue='Buyer Segment', palette=custom_palette, 
                          alpha=0.7, s=100)

plt.title('Real Estate Market Segmentation: Price vs. Bedrooms', fontsize=16, fontweight='bold')
plt.xlabel('Number of Bedrooms', fontsize=12)
plt.ylabel('Property Price (in Hundreds of Millions ₦)', fontsize=12)
plt.legend(title='Identified Segments')

# Format y-axis to show millions cleanly
def price_formatter(x, pos):
    return f'₦{x/1e6:.0f}M'
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(price_formatter))

plt.tight_layout()
plt.savefig('buyer_segments_clusters.png', dpi=300)