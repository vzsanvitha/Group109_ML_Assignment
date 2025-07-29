from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

# Load the best model from the "model" folder
model_path = os.path.join("model", "Logistic_Regression_best_model.pkl")
model = joblib.load(model_path)

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        if "features" not in data:
            return jsonify({"error": "Missing 'features' key in JSON"}), 400

        features = np.array(data["features"]).reshape(1, -1)
        prediction = model.predict(features)
        return jsonify({"prediction": int(prediction[0])})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7000,debug=True)
