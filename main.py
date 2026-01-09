from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ["*"]  # Allow all origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Must include all methods
    allow_headers=["*"],  # Must include all headers
)

# Load both the model and the encoders
model = pickle.load(open("student_model.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb")) # Load your saved encoders

@app.get("/")
def home():
    return {"message": "Student Exam Score API is running"}

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    
    # Use the LOADED encoders to transform data
    for col, le in encoders.items():
        if col in df.columns:
            # Use .transform(), NOT .fit_transform()
            df[col] = le.transform(df[col])
    
    prediction = model.predict(df)
    return {"predicted_exam_score": float(prediction[0])}

