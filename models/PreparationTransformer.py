from joblib import load

class Model:
    def __init__(self, columns):
        self.model = load("assets/pipeline1.joblib")

    def transform(self, data):
        result = self.model.transform(data)
        return result