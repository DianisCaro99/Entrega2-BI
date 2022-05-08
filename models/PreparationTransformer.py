from joblib import load

class PreparationTransformer:
    def __init__(self):
        self.model = load("assets/pipeline1.joblib")

    def transform(self, data):
        result = self.model.transform(data)
        return result