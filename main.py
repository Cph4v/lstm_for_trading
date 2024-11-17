import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from datetime import datetime, timedelta
from src.data_preprocessing import preprocess
from src.utils.postprocessing import plot, next_pred
from src.models.transformer import arima_forecast

# Initialize data storage
true_label = np.array([])
pred = np.array([])

fig, ax = plt.subplots(figsize=(10, 5))

def update(frame):
    global true_label, pred

    # Parameters and model path
    seq_len = 128
    rolling_factor = 5
    percent = 1
    model_path = '/home/amir/farshid_forex_env/farshid_forex_env/128seq_1min_ohlcv.keras'

    dataset_ex_df = preprocess(limit=1000)
    auto_arima(dataset_ex_df['high'], seasonal=False, trace=True)

    # Split data into train and test sets
    X = dataset_ex_df.values
    size = int(len(X) * 1)
    train, test = X[0:size], X[size:len(X)]

    history = [x for x in train]
    predictionss = list()

    yhat = arima_forecast(history)
    predictionss.append(yhat)

    pred = np.append(pred, yhat)
    true_label = np.append(true_label, history[-1][0])

    timestamps = [datetime.now() + timedelta(minutes=1*i) for i in range(len(true_label))]
    timestamps_pred = [datetime.now() + timedelta(minutes=1*i) for i in range(len(pred))]

    # Plotting
    ax.clear()
    ax.plot(timestamps, true_label, label='true_label', marker='x', color='blue')
    ax.plot(timestamps_pred, pred, color='brown', linewidth=2, label='predictions')
    plot(true_label, pred, None)

ani = FuncAnimation(fig, update, interval=50000)
plt.show()