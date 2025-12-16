import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# Load and prepare data
df = pd.read_csv(r"C:\Users\NITRO5\Desktop\4\used_cars.csv")

print("Before cleaning dtypes:")
print(df[['price', 'milage', 'model_year']].dtypes)


# Data cleaning
df['price'] = df['price'].replace(r'[\$,]', '', regex=True).astype(float)
df['milage'] = df['milage'].astype(str).str.replace('mi.', '', regex=False)
df['milage'] = df['milage'].str.replace(',', '', regex=False)
df['milage'] = pd.to_numeric(df['milage'], errors='coerce')
df['model_year'] = pd.to_numeric(df['model_year'], errors='coerce')

# Fill missing values
df['milage'].fillna(df['milage'].median(), inplace=True)
df['model_year'].fillna(df['model_year'].median(), inplace=True)
df['fuel_type'].fillna('Unknown', inplace=True)
df['engine'].fillna('Unknown', inplace=True)
df['transmission'].fillna('Unknown', inplace=True)
df['accident'].fillna('None reported', inplace=True)
df['clean_title'].fillna('Unknown', inplace=True)

nan_p = (df.isna().sum())
print('nan ',nan_p)


# After cleaning, verify
print("\nAfter cleaning dtypes:")
print(df[['price', 'milage', 'model_year']].dtypes)


import re

def extract_hp(engine_text):
    if isinstance(engine_text, str):
        match = re.search(r'(\d+(?:\.\d+)?)\s*HP', engine_text, re.IGNORECASE)
        if match:
            return float(match.group(1))
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', engine_text)
        if numbers:
            return float(max(numbers, key=float))
    return np.nan

def extract_cylinders(engine_text):
    if isinstance(engine_text, str):
        patterns = [(r'V(\d+)', 1), (r'I(\d+)', 1), (r'(\d+)\s*Cylinder', 1)]
        for pattern, group in patterns:
            match = re.search(pattern, engine_text, re.IGNORECASE)
            if match:
                try:
                    return int(match.group(group))
                except:
                    continue
    return np.nan

def extract_engine_liters(engine_text):
    if isinstance(engine_text, str):
        match = re.search(r'(\d+\.\d+)\s*L', engine_text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except:
                pass
    return np.nan

# Apply feature extraction
df['engine_hp'] = df['engine'].apply(extract_hp)
df['engine_hp'].fillna(df['engine_hp'].median(), inplace=True)

df['cylinders'] = df['engine'].apply(extract_cylinders)
print(df['cylinders'].median())
df['cylinders'].fillna(4, inplace=True)

df['engine_liters'] = df['engine'].apply(extract_engine_liters)
df['engine_liters'].fillna(df['engine_liters'].median(), inplace=True)

df['has_turbo'] = df['engine'].apply(lambda x: 1 if isinstance(x, str) and 'turbo' in x.lower() else 0)
df['is_hybrid_electric'] = df['engine'].apply(lambda x: 1 if isinstance(x, str) and ('hybrid' in x.lower() or 'electric' in x.lower()) else 0)

# Calculate car age
current_year = 2024
df['car_age'] = current_year - df['model_year']
df['car_age'] = df['car_age'].apply(lambda x: max(0, x))

# إزالة القيم المتطرفة
df = df[df['price'] <= 150000].copy()

# Encode categorical variables
# categorical_cols = ['brand', 'fuel_type', 'transmission', 'accident', 'clean_title']
categorical_cols = ['brand', 'fuel_type', 'transmission', 'accident']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Define features
features = [
    'model_year', 'milage', 'car_age', 'engine_hp', 'cylinders',
    'engine_liters', 'has_turbo', 'is_hybrid_electric',
    'brand_encoded', 'fuel_type_encoded', 'transmission_encoded',
    'accident_encoded'
]

X = df[features]
y = df['price']

# تحقق من أن X و y لها نفس الطول
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Number of NaN in X: {X.isna().sum().sum()}")
print(f"Number of NaN in y: {y.isna().sum()}")

# Remove any remaining NaN values
X = X.fillna(X.median())

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape}")
print(f"Testing set size: {X_test.shape}")
print(f"Training target size: {y_train.shape}")
print(f"Testing target size: {y_test.shape}")

# train
# تعريف النموذج
model = RandomForestRegressor(n_estimators=350, random_state=42)

print(f"Training Random Forest (350 trees)...")

try:
    # تدريب النموذج
    model.fit(X_train, y_train)
    
    # عمل التنبؤات
    y_pred = model.predict(X_test)
    
    # حساب المقاييس
    r2 = r2_score(y_test, y_pred)


    # عرض النتائج
    print(f"\n=== Random Forest (n_estimators=350) Results ===")
    print(f"  R² Score: {r2:.4f}")
    
    import pickle
    import joblib

    # حفظ النموذج والـ label encoders
    model_data = {
        'model': model,
        'label_encoders': label_encoders,
        'features': features,
        'X_train_columns': X_train.columns.tolist()  # حفظ ترتيب الأعمدة
    }

    # حفظ الملف
    with open('random_forest_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)

    print("✅ Model saved successfully to 'random_forest_model.pkl'")
    print("🎯 You can now run the FastAPI server!")   

    
    # تخزين النتائج
    results = {
        'Model': 'Random Forest (350 trees)',
        'R²': r2,
        'Predictions': y_pred
    }
    
except Exception as e:
    print(f"  Error: {str(e)}")


    