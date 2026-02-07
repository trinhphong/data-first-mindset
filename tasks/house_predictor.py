import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 1. DATA INGESTION (Pandas)
data = {
    "sqft": [1500, 1800, 2400, 3000, 3500, 1200, 5000, 2200, 2800, 4200],
    "bedrooms": [2, 3, 3, 4, 4, 5, 5, 6, 6, 7],
    "price": [
        300000,
        340000,
        450000,
        550000,
        620000,
        250000,
        950000,
        410000,
        510000,
        800000,
    ],
}
df = pd.DataFrame(data)

# 2. FEATURE SELECTION
# X = Input (Matrix), y = Target (Vector)
X = df[["sqft", "bedrooms"]]
y = df["price"]

# 3. THE "TRAIN/TEST SPLIT" (A key AI concept)
# We hide 20% of data from the AI to test it later (like an exam)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. MODEL TRAINING
model = LinearRegression()
model.fit(X_train, y_train)  # This is where the "Learning" happens


# 5. PREDICTION (The API endpoint logic)
def predict_price(sqft_input, bedroom_input):
    prediction = model.predict([[sqft_input, bedroom_input]])
    return prediction[0]


# Test it
square = 1500
bedroom = 5
estimated_price = predict_price(square, bedroom)
print(
    f"A {square} sqft house with {bedroom} bedrooms is estimated to cost: ${estimated_price:,.2f}"
)
