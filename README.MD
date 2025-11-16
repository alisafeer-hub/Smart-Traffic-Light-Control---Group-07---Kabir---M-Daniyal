    🚦 Smart Traffic Light Prediction System
Machine Learning Based Traffic Signal Optimizer (Random Forest Model)
This project predicts which signal should turn green next based on:
•	Number of cars in each lane (North, East, West, South)
•	Arrival rate (minutes per car)
The system uses a trained Random Forest model to identify the lane with highest urgency.
________________________________________
📌 1. Requirements
Before running the project, install the following:
✔ Python Version
Python 3.10 or Python 3.11
________________________________________
📌 2. Required Libraries
These libraries are used in your project:
Core Libraries
Library	Purpose
numpy	Numeric operations
pandas	Dataset loading & preprocessing
scikit-learn	ML models (Random Forest, DNN, etc.)
matplotlib	Accuracy visualization
joblib	Save & load ML models
flask	Web backend for prediction
json	Data handling in API
warnings	Ignore model warnings
________________________________________
📌 3. Install All Libraries
Copy–paste this in CMD / VS Code terminal:
pip install numpy pandas scikit-learn matplotlib joblib flask
If you get "pip not recognized", use:
python -m pip install --upgrade pip
________________________________________
📌 4. Project Structure
Smart-traffic/
│── app.py                  # Flask backend
│── model.pkl               # Trained Random Forest Model
│── index.html              # Frontend input page
│── static/
│     └── style.css         # Frontend design
│── README.md               # Documentation
________________________________________
📌 5. Running the Project
Step 1 — Train the Model (Only once)
Use your RF training script:
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import joblib

df = pd.read_csv("traffic_data.csv")

X = df.drop("next_green", axis=1)
y = df["next_green"]

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, "model.pkl")
This saves the trained AI model.
________________________________________
📌 Step 2 — Run Flask Backend
Open terminal inside project folder:
python app.py
You will see:
* Running on http://127.0.0.1:5000
Open browser → paste the link.
________________________________________
📌 Step 3 — Use Web App
In the UI, enter:
•	Cars in North / East / West / South
•	Arrival rate (minutes per car)
•	Current green signal
Then click Predict Next Green.
Backend will calculate:
•	Traffic load
•	Future incoming traffic
•	Lane urgency
•	Predicted next green signal
•	Traffic density (Low / Medium / High / Extreme)
________________________________________
📌 6. Random Forest Prediction Logic (Simplified)
Your model finds:
Future Traffic = current_cars + (60 / arrival_rate)
Then the model chooses the lane with:
✔ highest load
✔ most incoming cars
✔ lowest waiting time
________________________________________
📌 7. How To Modify the Model
Retrain using the Kaggle dataset:
Kaggle link used:
https://www.kaggle.com/datasets/fedesoriano/traffic-prediction-dataset
Load and merge into your system:
df = pd.read_csv("traffic.csv")
Then split → train → save model with joblib.
________________________________________
📌 8. API Route Summary
POST /predict
Input fields:
{
  "north": 4,
  "east": 3,
  "west": 6,
  "south": 2,
  "north_rate": 10,
  "east_rate": 12,
  "west_rate": 5,
  "south_rate": 15,
  "current_green": 1
}
Backend returns:
{
  "predicted_signal": "North",
  "density": "High"
}
________________________________________
📌 9. Traffic Density Logic
0–20   → Low  
21–40  → Medium  
41–70  → High  
70+    → Extreme  
________________________________________
📌 10. Final Notes
•	Random Forest produced highest accuracy: 98.2%
•	Faster than DNN & RL in your dataset
•	Works best for real-time prediction

