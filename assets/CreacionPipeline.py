import joblib
from sklearn.pipeline import Pipeline
from PipelineInicial import LimpiezaTransformer
from PipelineInicial import NormalizacionTransformer


pipe1 = Pipeline(steps=[('limpieza', LimpiezaTransformer()),
                       ('normalizar', NormalizacionTransformer())])

joblib.dump(pipe1, 'pipeline.pkld')