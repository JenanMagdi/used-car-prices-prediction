from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np
import uvicorn

# تعريف بيانات الإدخال
class CarFeatures(BaseModel):
    brand: str
    model_year: int
    milage: float
    engine_hp: float
    cylinders: int
    engine_liters: float
    fuel_type: str
    transmission: str
    accident: str
    has_turbo: int = 0
    is_hybrid_electric: int = 0

# تحميل النموذج
try:
    with open('random_forest_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    label_encoders = model_data['label_encoders']
    features = model_data['features']
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# إنشاء التطبيق
app = FastAPI()

# إعداد CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Car Price Prediction API is running!"}

@app.post("/predict")
def predict(car: CarFeatures):
    try:
        # حساب عمر السيارة
        car_age = 2024 - car.model_year
        
        # إعداد بيانات الإدخال
        input_data = {}
        
        # القيم العددية
        input_data['model_year'] = car.model_year
        input_data['milage'] = car.milage
        input_data['engine_hp'] = car.engine_hp
        input_data['cylinders'] = car.cylinders
        input_data['engine_liters'] = car.engine_liters
        input_data['has_turbo'] = car.has_turbo
        input_data['is_hybrid_electric'] = car.is_hybrid_electric
        input_data['car_age'] = car_age
        
        # القيم الفئوية مع قيم افتراضية
        categorical_values = {
            'brand': car.brand,
            'fuel_type': car.fuel_type,
            'transmission': car.transmission,
            'accident': car.accident,
        }
        
        # تشفير القيم الفئوية
        for col, value in categorical_values.items():
            le = label_encoders[col]
            # إذا كانت القيمة غير موجودة، استخدم القيمة الأولى
            if str(value) not in le.classes_:
                value = le.classes_[0]
            else:
                value = str(value)
            input_data[f'{col}_encoded'] = le.transform([value])[0]
        
        # ترتيب الميزات كما في التدريب
        feature_vector = []
        for feature in features:
            feature_vector.append(input_data[feature])
        
        # التنبؤ
        prediction = model.predict([feature_vector])[0]
        
        return {
            "status": "success",
            "predicted_price": round(float(prediction), 2)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("🚀 Starting FastAPI server on http://127.0.0.1:8000")
    print("📝 Open http://127.0.0.1:8000 in browser to test")
    uvicorn.run(app, host="127.0.0.1", port=8000)