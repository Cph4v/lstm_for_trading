import tensorflow as tf
from .layers import Time2Vector, SingleAttention, MultiAttention, TransformerEncoder
from statsmodels.tsa.arima.model import ARIMA

def load_transformer_model(model_path):
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            'Time2Vector': Time2Vector,
            'SingleAttention': SingleAttention,
            'MultiAttention': MultiAttention,
            'TransformerEncoder': TransformerEncoder,
        }
    )
    return model


def arima_forecast(history):
    model = ARIMA(history, order=(0,1,0))
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    return yhat