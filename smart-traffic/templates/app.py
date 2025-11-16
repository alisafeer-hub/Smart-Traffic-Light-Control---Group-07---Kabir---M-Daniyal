# -----------------------------
# Smart Traffic Light API (Flask)
# -----------------------------

from flask import Flask, request, render_template
from flask import render_template

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

app = Flask(__name__)

# -----------------------------
# Load Dataset & Train Model
# -----------------------------
df = pd.read_csv("smart_traffic_dataset.csv")  # CSV must be in same folder

# Features & Target
X = df[["intersection_id", "lane_id", "arrival_rate", "avg_wait_time", "total_vehicles", "signal_timer"]]
y = df["congestion_level"]

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y_encoded)

# Map congestion level to recommended green time (seconds)
congestion_to_green = {
    "Low": 30,
    "Medium": 60,
    "High": 90,
    "Very High": 120,
    "Extreme": 150
}

# -----------------------------
# Root route (show HTML form)
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    level = None
    form_data = {}
    
    if request.method == "POST":
        # Get form data
        form_data = {
            "north": int(request.form.get("north", 0)),
            "east": int(request.form.get("east", 0)),
            "west": int(request.form.get("west", 0)),
            "south": int(request.form.get("south", 0)),
            "north_rate": int(request.form.get("north_rate", 0)),
            "east_rate": int(request.form.get("east_rate", 0)),
            "west_rate": int(request.form.get("west_rate", 0)),
            "south_rate": int(request.form.get("south_rate", 0)),
            "current_green": int(request.form.get("current_green", 1))
        }

        # For demonstration, just use current_green as prediction
        prediction = form_data["current_green"]
        
        # Calculate traffic level based on total cars
        total_cars = form_data["north"] + form_data["east"] + form_data["west"] + form_data["south"]
        if total_cars <= 10:
            level = "Low"
        elif total_cars <= 20:
            level = "Medium"
        elif total_cars <= 30:
            level = "High"
        else:
            level = "Extreme"

    return render_template("index.html", prediction=prediction, level=level, form_data=form_data)

# -----------------------------
# Run Flask App
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
