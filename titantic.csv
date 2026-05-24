# decision_tree_bank.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Step 1: Load dataset
# Download the Bank Marketing dataset (bank.csv) from UCI ML Repository
df = pd.read_csv("bank.csv", sep=";")

# Step 2: Inspect data
print("Dataset Shape:", df.shape)
print("Columns:", df.columns)

# Step 3: Encode categorical variables
# Convert categorical columns into numeric using one-hot encoding
df_encoded = pd.get_dummies(df, drop_first=True)

# Step 4: Split features and target
X = df_encoded.drop("y_yes", axis=1)   # Features
y = df_encoded["y_yes"]                # Target (1 = purchase, 0 = no purchase)

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Step 6: Build Decision Tree Classifier
clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(X_train, y_train)

# Step 7: Predictions
y_pred = clf.predict(X_test)

# Step 8: Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 9: Visualize Decision Tree
plt.figure(figsize=(20,10))
plot_tree(clf, feature_names=X.columns, class_names=["No", "Yes"], filled=True)
plt.show()
