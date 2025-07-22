# 👥 Customer Personality Analysis

This project analyzes customer personality data to segment consumers based on demographics and purchasing behavior. The goal is to derive meaningful clusters and identify patterns to support better business decisions and enable targeted marketing strategies.

---

## 📁 Project Structure

- `Cleaned_CustomerPersonalityData.csv` – Preprocessed dataset used for clustering analysis  
- `Project2.py` – Python script containing data preprocessing, clustering models, and final evaluation  
- `Project_2.ipynb` – Jupyter Notebook version for exploratory data analysis, visualization, and modeling  

---

## 🧪 Objectives

- Understand customer purchasing behavior through unsupervised learning  
- Engineer meaningful features like Age, Household Size, Spending, Parenthood, and Relationship Status  
- Apply clustering algorithms to segment customers effectively and interpret the clusters  

---

## 🛠️ Methods and Tools

**Languages & Libraries:**
- Python  
- Pandas, NumPy  
- Scikit-learn  
- Matplotlib, Seaborn  

**Techniques:**
- Data Cleaning & Feature Engineering  
- Standardization & PCA for dimensionality reduction  
- Clustering:
  - KMeans  
  - Gaussian Mixture Model (GMM)  
  - Hierarchical Clustering  
- Evaluation Metrics:
  - Silhouette Score  
  - Bayesian Information Criterion (BIC)  

---

## 📊 Key Features Engineered

- `Age` – Derived from birth year  
- `Membership_Duration` – Number of years since customer joined  
- `Spending_Total` – Combined spending across all product categories  
- `HouseHoldSize` – Based on relationship status and number of children  
- `Parent` – Boolean flag indicating if the customer is a parent  

---

## 📈 Results Summary

- Optimal clustering achieved using GMM with 9 clusters (lowest BIC score)  
- Final segment definitions included:
  - High-Income Parents  
  - Low-Income Parents  
  - High-Income Non-Parents  
  - Low-Income Non-Parents  
  - Elders  

---

## 📂 Dataset

The full dataset is available on **Kaggle**:  
👉 [Customer Personality Analysis Dataset](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis)

---

## 👨‍💻 Author

**Jared Gonzalez**, 
**A Sai Prasanth Reddy**,
**Tejaswi Chigurupati**,

California State University, San Bernardino
Bachelors in Science, Computer Science
Minor in Data Science  

---

## 📃 License

This project is for academic purposes and is shared under the MIT License.
