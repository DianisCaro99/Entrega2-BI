from joblib import load

class Model:

    def __init__(self,columns):
        self.model = load("assets/svcmodel.joblib")

    def predict(self, data):
        result = self.model.predict(data)
        return result