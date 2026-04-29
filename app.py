from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# load model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = [float(x) for x in request.form.values()]
        final_features = np.array(features).reshape(1, -1)

        prediction = model.predict(final_features)

        result = "Loan Approved" if prediction[0] == 1 else "Loan Rejected"

        return render_template("index.html", prediction_text=result)
    
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)