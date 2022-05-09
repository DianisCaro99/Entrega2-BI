from flask import Flask, request, jsonify
from flask_cors import CORS
from joblib import load
import nltk
import pandas as pd
import PipelineInicial

app = Flask(__name__)
cors = CORS(app)

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

pipe1 = load("assets/pipeline.pkld")
vectorizer = load('assets/vectorizer.joblib')
model = load("assets/svcmodel.joblib")

@app.get("/api")
def read_root():
   return "Entrega 2 - Grupo 5: Automatización analítica de textos"

@app.route("/api/prediction", methods=["POST"])
def make_predictions_r():
    data = request.get_data().decode('utf-8')
    df = pd.read_json(data)
    registrotrans = pipe1.transform(df)
    registrotrans = vectorizer.transform(registrotrans['words'])
    prediction = model.predict(registrotrans)[0]
    proba = model.predict_proba(registrotrans)[0]
    if prediction:
        proba = round(proba[1],3)
    else:
        proba = round(proba[0],3)

    return jsonify(response=['Eligible' if prediction == 1 else 'Not eligible',proba])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
