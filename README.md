# Customer Personality data
This project analyzes customer personality data to segment consumers based on demographics and purchasing behavior. The goal is to derive meaningful clusters and identify patterns for better business decision-making and targeted marketing.

📁 Project Structure

Cleaned_CustomerPersonalityData.csv: Preprocessed dataset used for analysis.
Project2.py: Python script version of the project containing data preprocessing, clustering, and analysis code.
Project_2.ipynb: Jupyter Notebook version for exploratory data analysis, model training, and visualization.
🧪 Objectives

Understand customer purchasing behavior through clustering.
Engineer meaningful features such as Age, Household Size, Spending, Parenthood, and Relationship Status.
Apply clustering algorithms to segment customers effectively.
🛠️ Methods and Tools

Languages & Libraries: Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
Techniques:
Data Cleaning & Feature Engineering
Standardization & Dimensionality Reduction (PCA)
Clustering: KMeans, Gaussian Mixture Model (GMM), Hierarchical Clustering
Evaluation: Silhouette Score, BIC
📊 Key Features Engineered

Age: Derived from birth year.
Membership_Duration: Number of years since customer joined.
Spending_Total: Sum of all product category spending.
HouseHoldSize: Computed from relationship status and number of children.
Parent: Flag indicating parenthood status.
📈 Results Summary

Achieved optimal clustering using GMM with 9 clusters (lowest BIC score).
Final clusters included:
High-Income Parents
Low-Income Parents
High-Income Non-Parents
Low-Income Non-Parents
Elders

For the full dataset please click this link for kaggle: [Customer Personality Data](https://www.kaggle.com/code/karnikakapoor/customer-segmentation-clustering) 
