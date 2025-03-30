# prodigy_task_3
this is one of the task given by prodigy 
# Import necessary libraries  
import pandas as pd  
from sklearn.model_selection import train_test_split  
from sklearn.tree import DecisionTreeClassifier  
from sklearn.metrics import classification_report, accuracy_score  
import matplotlib.pyplot as plt  
from sklearn.tree import plot_tree  

# Step 1: Load the dataset  
data = pd.read_csv('bank.csv', sep=';')  

# Step 2: Preprocess the data  
data = pd.get_dummies(data, drop_first=True)  # Convert categorical variables  

# Define features and target variable  
X = data.drop('y_yes', axis=1)  # Replace 'y_yes' with the actual target column name  
y = data['y_yes']  # Replace 'y_yes' with the actual target column name  

# Step 3: Split the dataset into training and testing sets  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)  

# Step 4: Build the Decision Tree Classifier  
classifier = DecisionTreeClassifier(random_state=42)  
classifier.fit(X_train, y_train)  

# Step 5: Evaluate the model  
y_pred = classifier.predict(X_test)  
print(classification_report(y_test, y_pred))  
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')  

# Step 6: Visualize the Decision Tree  
plt.figure(figsize=(12, 8))  
plot_tree(classifier, filled=True, feature_names=X.columns, class_names=['No', 'Yes'])  
plt.title("Decision Tree Classifier")  
plt.show()  
