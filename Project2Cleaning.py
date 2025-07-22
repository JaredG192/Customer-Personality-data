import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# Loading the data
try:
    data = pd.read_csv("CustomerPersonalityAnalysisData.csv")
    print("CSV loaded successfully!")
    print(data.head())  # Show the first few rows of the dataset
    print("Columns in the DataFrame:", data.columns.tolist())

except FileNotFoundError:
    print("CSV file not found. Please check the file path.")
except Exception as e:
    print(f"An error occurred: {e}")

print("Number of datapoints:", len(data))
data.head()
data.info()

#visualize data before cleaning
custom_palette = sns.color_palette("viridis", 3)
sns.pairplot(data, vars=['Year_Birth', 'Income', 'MntWines'], hue='Kidhome', palette=custom_palette, diag_kind='kde', markers = ["o", "s", "D"], height = 2)
plt.show()


#Cleaning the data set
#1. Remove missing values in 'Income'
data = data.dropna(subset='Income')


#2. Convert 'Dt_Customer' to datetime format
data['Dt_Customer'] = pd.to_datetime(data['Dt_Customer'], format='%d-%m-%Y', errors='coerce')


#3. Create a new column membership to find how long a customer has been a member and replace 'Dt_Customer'
good_dates = data['Dt_Customer'].dropna()
if not good_dates.empty:
    newest_enrollment_date = good_dates.max()
    oldest_enrollment_date = good_dates.min()

    print("Newest enrollment date in the record: ", newest_enrollment_date)
    print("Oldest enrollment date in the record: ", oldest_enrollment_date)
data['Membership'] = (newest_enrollment_date - data['Dt_Customer']).dt.days


#4. Make a column for age to replace 'Year_Birth'
data["Age"] = 2024 - data["Year_Birth"]


#5. Make a column for total spent
data["TotalSpent"] = data["MntWines"]+ data["MntFruits"]+ data["MntMeatProducts"]+ data["MntFishProducts"]+ data["MntSweetProducts"]+ data["MntGoldProds"]


#6. Make a column for relationship status to replace 'Martial_Staus'
data["RelationshipStatus"] = data["Marital_Status"].replace({"Married": "Partner", "Together": "Partner", "Absurd": "Alone", "Widow": "Alone", "YOLO": "Alone", "Divorced": "Alone", "Single": "Alone"})


#7. Combine and replace 'kidhome' and 'teenhome' into one column 'children'
data["Children"]= data["Kidhome"]+ data["Teenhome"]


#8. Make column for Household size
data["RelationshipStatusInt"] = data["RelationshipStatus"].map({"Alone": 1, "Partner": 2})
data["HouseHoldSize"] = data["RelationshipStatusInt"].astype(int) + data["Children"]

#9. Make Column for who is a parent or not
data["Parent"] = np.where(data["Children"] > 0, 1, 0)


#10. combine Education into 3 groups: Undergraduate, Gradate, and Postgraduate
data["Education"]=data["Education"].replace({"Basic":"Undergraduate","2n Cycle":"Undergraduate", "Graduation":"Graduate", "Master":"Postgraduate", "PhD":"Postgraduate"})


#11. Make columns easier to read
data = data.rename(columns = {"MntWines": "Wines","MntFruits":"Fruits","MntMeatProducts":"Meat","MntFishProducts":"Fish","MntSweetProducts":"Sweets","MntGoldProds":"Gold"})


#12. Remove any outliers in 'Income' and 'Age'
income_limit= data['Income'].quantile(0.99)
data = data[data['Income'] <= income_limit]
age_limit = data['Age'].quantile(0.99)
data = data[data['Age'] <= age_limit]


#13. Remove any unnecesary coulmns
data.drop(columns=['Z_CostContact', 'Z_Revenue', 'Marital_Status', 'Dt_Customer', 'Year_Birth', 'ID', 'Teenhome', 'Kidhome'], inplace=True)


#Check cleaned dataset
print(data.describe())

#visualize data after cleaning
custom_palette = sns.color_palette(["#FF6347", "#4682B4"])
sns.pairplot(data, vars=['Income', 'Membership', 'Age', 'TotalSpent', 'HouseHoldSize'],
hue='Parent', palette=custom_palette, diag_kind='kde', markers=["o", "s"], height = 2)
plt.show()
