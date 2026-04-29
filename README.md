# Segmenting the Nigerian Real Estate Market: A Data-Driven Approach

**Course:** CS 4412: Data Mining  

## 📌 Project Overview
The Nigerian real estate market is rapidly expanding, generating vast amounts of pricing, locational, and physical amenity data. This project leverages unsupervised machine learning (K-Means Clustering) to transition from broad, untargeted property marketing to precise, data-driven buyer segmentation. 

By analyzing over 1,600 property listings scraped from the Nigeria Property Centre, this analysis uncovers three distinct buyer tiers and isolates the specific geographic features that drive exponential premium pricing.

### 🎯 Discovery Questions
1. Can we identify distinct, underserved buyer segments based purely on property features and price?
2. What underlying physical or locational features truly drive premium pricing in the housing market?

## 📊 Key Findings
* **Tier 1 (Starter / Investor):** Averages ₦138M | ~2.5 Beds. Represents compact apartments and semi-detached units.
* **Tier 2 (Mid-Market Family):** Averages ₦215M | ~4.2 Beds. The bulk of the market, driven by a baseline expectation for en-suite facilities.
* **Tier 3 (Premium Luxury):** Averages ₦601M | ~5.2 Beds. Expanding past 4 bedrooms does not exponentially increase price—*location* does. 82% of these properties are concentrated in Lekki, Ikeja, Magodo, Ikoyi, and Guzape District.

*Validation: Clusters were mathematically validated using a Silhouette Score of 0.417.*

## 📂 Repository Structure
```text
├── data/
│   └── crib.csv                 # Raw dataset (Nigeria Property Centre listings)
├── notebooks/
│   └── clustering_analysis.ipynb # Jupyter notebook with EDA and K-Means modeling
├── src/
│   └── model.py                 # Core Python script for data preprocessing and clustering
├── figures/
│   └── buyer_segments.png       # Exported visualizations for reports/slides
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation