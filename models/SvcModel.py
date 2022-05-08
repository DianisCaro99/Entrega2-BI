from joblib import load

class Model:
    def __init__(self):
        self.model = load("assets/svcmodel.joblib")

    def predict(self, data):
        result = self.model.predict(data)
        return result
    
    def predict_proba(self, data):
        result = self.model.predict_proba(data)
        return result