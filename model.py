import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

    # Load dataset
df = pd.read_csv("cleaned_dataset.csv")
print("✅ Dataset loaded")

# Convert VisitorType
df['VisitorType'] = df['VisitorType'].replace({'Returning_Visitor': 1,'New_Visitor': 0,'Other': 0})

    # Drop Month column
if 'Month' in df.columns:
  df = df.drop(['Month'], axis=1)

    # Features & target
X = df.drop('Revenue', axis=1)
y = df['Revenue']

    # Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

    # Save model
pickle.dump(model, open("model.pkl", "wb"))
print("🎉 Model trained and saved successfully!")