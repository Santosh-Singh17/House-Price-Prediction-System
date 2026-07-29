# Import Libraries
import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load Dataset
df = pd.read_csv("Housing.csv")

# Fill Missing Values
df.fillna(df.mode().iloc[0], inplace=True)

# Remove Duplicate Values
df.drop_duplicates(inplace=True)

# Encode Categorical Columns
encoder = LabelEncoder()

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = encoder.fit_transform(df[col])

# Separate Features and Target
X = df.drop("price", axis=1)
y = df["price"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Random Forest Model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Save Model
joblib.dump(model, "house_price_model.pkl")

print("Model Trained Successfully!")
print("house_price_model.pkl has been created.")