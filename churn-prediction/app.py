from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load the dataset from CSV file
df = pd.read_csv("E Commerce new.csv")

# Fill missing values with mean for numeric columns
numeric_columns = [
    'Tenure', 'WarehouseToHome', 'HourSpendOnApp', 'OrderAmountHikeFromlastYear', 
    'CouponUsed', 'OrderCount', 'DaySinceLastOrder'
]
for col in numeric_columns:
    df[col].fillna(df[col].mean(), inplace=True)

# One-Hot Encoding for categorical variables
categorical_columns = ['Gender', 'PreferredLoginDevice', 'PreferedOrderCat', 'PreferredPaymentMode', 'MaritalStatus']
df = pd.get_dummies(data=df, columns=categorical_columns, drop_first=True)

# Define feature columns
feature_columns = [
    'Tenure', 'CityTier', 'WarehouseToHome', 'HourSpendOnApp', 'NumberOfDeviceRegistered', 
    'SatisfactionScore', 'NumberOfAddress', 'Complain', 'OrderAmountHikeFromlastYear', 
    'CouponUsed', 'OrderCount', 'DaySinceLastOrder', 'CashbackAmount'
]
# Add dynamically created columns from one-hot encoding
feature_columns += [col for col in df.columns if col.startswith(('Gender_', 'PreferredLoginDevice_', 'PreferedOrderCat_', 'PreferredPaymentMode_', 'MaritalStatus_'))]

X = df[feature_columns]
y = df['Churn']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = LogisticRegression(max_iter=2000)
model.fit(X_train_scaled, y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    input_data = {
        'Tenure': float(request.form['tenure']),
        'CityTier': int(request.form['city_tier']),
        'WarehouseToHome': float(request.form['warehouse_to_home']),
        'HourSpendOnApp': float(request.form['hour_spend_on_app']),
        'NumberOfDeviceRegistered': int(request.form['num_of_devices_registered']),
        'SatisfactionScore': float(request.form['satisfaction_score']),
        'NumberOfAddress': int(request.form['num_of_addresses']),
        'Complain': int(request.form['complain']),
        'OrderAmountHikeFromlastYear': float(request.form['order_amount_hike']),
        'CouponUsed': int(request.form['coupon_used']),
        'OrderCount': int(request.form['order_count']),
        'DaySinceLastOrder': int(request.form['days_since_last_order']),
        'CashbackAmount': float(request.form['cashback_amount']),
        'Gender_Male': 1 if request.form['gender'] == 'male' else 0,
        'PreferredLoginDevice_Computer': 1 if request.form['preferred_login_device'] == 'computer' else 0,
        'PreferredLoginDevice_Mobile Phone': 1 if request.form['preferred_login_device'] == 'mobile phone' else 0,
        'PreferredLoginDevice_Phone': 1 if request.form['preferred_login_device'] == 'phone' else 0,
        'PreferedOrderCat_Fashion': 1 if request.form['preferred_order_cat'] == 'fashion' else 0,
        'PreferedOrderCat_Grocery': 1 if request.form['preferred_order_cat'] == 'grocery' else 0,
        'PreferedOrderCat_Laptop & Accessory': 1 if request.form['preferred_order_cat'] == 'laptop & accessory' else 0,
        'PreferedOrderCat_Mobile': 1 if request.form['preferred_order_cat'] == 'mobile' else 0,
        'PreferedOrderCat_Mobile Phone': 1 if request.form['preferred_order_cat'] == 'mobile phone' else 0,
        'PreferedOrderCat_Others': 1 if request.form['preferred_order_cat'] == 'others' else 0,
        'PreferredPaymentMode_CC': 1 if request.form['preferred_payment_mode'] == 'cc' else 0,
        'PreferredPaymentMode_COD': 1 if request.form['preferred_payment_mode'] == 'cod' else 0,
        'PreferredPaymentMode_Cash on Delivery': 1 if request.form['preferred_payment_mode'] == 'cash on delivery' else 0,
        'PreferredPaymentMode_Credit Card': 1 if request.form['preferred_payment_mode'] == 'credit card' else 0,
        'PreferredPaymentMode_Debit Card': 1 if request.form['preferred_payment_mode'] == 'debit card' else 0,
        'PreferredPaymentMode_E wallet': 1 if request.form['preferred_payment_mode'] == 'e wallet' else 0,
        'PreferredPaymentMode_UPI': 1 if request.form['preferred_payment_mode'] == 'upi' else 0,
        'MaritalStatus_Divorced': 1 if request.form['marital_status'] == 'divorced' else 0,
        'MaritalStatus_Married': 1 if request.form['marital_status'] == 'married' else 0,
        'MaritalStatus_Single': 1 if request.form['marital_status'] == 'single' else 0
    }

    # Convert input data to DataFrame
    test_df = pd.DataFrame([input_data])

    # Add missing columns with default value 0
    for col in X_train.columns:
        if col not in test_df.columns:
            test_df[col] = 0

    # Reorder columns to match the training data
    test_df = test_df[X_train.columns]

    # Scale the test data
    test_df_scaled = scaler.transform(test_df)

    # Predict churn for test person
    predicted_churn = model.predict(test_df_scaled)[0]
    
    return render_template('result.html', predicted_churn=predicted_churn)

if __name__ == '__main__':
    app.run(debug=True)
