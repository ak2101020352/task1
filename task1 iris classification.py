#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Step 1: Import Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Step 2: Load the Dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target

# Step 3: Explore the Dataset
print(df.head())
print(df.describe())
print(df['species'].value_counts())

# Visualize the dataset
sns.pairplot(df, hue='species', markers=['o', 's', 'D'])
plt.show()

# Step 4: Prepare the Data
X = df.drop(columns=['species'])
y = df['species']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 5: Train the Model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Step 6: Evaluate the Model
y_pred = knn.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Step 7: Make Predictions
# Example: Predicting the species of a new iris flower
new_sample = [[5.1, 3.5, 1.4, 0.2]]  # Measurements of a new flower
new_sample_scaled = scaler.transform(new_sample)
prediction = knn.predict(new_sample_scaled)
print(f'Predicted species: {iris.target_names[prediction[0]]}')


# In[ ]:




