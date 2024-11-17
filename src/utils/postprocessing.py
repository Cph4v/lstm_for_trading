import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta



def postprocessing(train_pred, max_return, min_return, before_pct_changes, prediction_index=-1):
    '''
    y_true[i+1] = y_true[i] - (train_pred[i] * y_true[i]) 
    '''
    label_true_denormalized = train_pred.copy()
    label_true_denormalized = label_true_denormalized*(max_return - min_return) + min_return


    label_true_denormalized = np.array(label_true_denormalized)
    predictions = label_true_denormalized.copy()
    predictions = predictions.flatten()

    y_true = before_pct_changes['high'].values
    y_true_1 = y_true[abs(len(label_true_denormalized) - len(y_true)):]

    y_true_1 = np.array(y_true_1)
    true_label = y_true_1.copy()
    true_label = true_label.flatten()

    # five_min_pred = true_label[prediction_index] - (predictions[prediction_index] * true_label[prediction_index])
    
    five_min_pred = (1 + predictions[prediction_index]).cumprod()
    five_min_pred *= true_label[prediction_index]

    last_5min = true_label[prediction_index]

    y_true_reconst = []
    for i in range(0,len(true_label)):
        y_true_reconst.append(true_label[i] - (predictions[i] * true_label[i]))

    # y_true_reconst.append(five_min_pred)
    
    return five_min_pred, last_5min, true_label, y_true_reconst      


def plot(true_label, predictions, before_pct_changes):
    num_labels = len(true_label)
    base_time = datetime.now()
    timestamps = [base_time + timedelta(minutes=5 * i) for i in range(num_labels)]

    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, true_label, label='true_label', marker='x', color='blue')
    plt.scatter([timestamps[-1]], [true_label[-1]], color='red', label='Last Label', s=100)

    for i, txt in enumerate(true_label):
        plt.annotate(f'{txt:.2f}', (timestamps[i], true_label[i]), textcoords="offset points", xytext=(0,10), ha='center')

    if len(true_label) > 1:
        plt.plot(timestamps[-2:], true_label[-2:], color='green', linewidth=2, label='Special Connection')

    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Values')
    plt.title('Label Values with Highlight on Last Item')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def next_pred(train_pred, data):
    dff = data.copy()
    tr = train_pred.copy()

    tr = tr[:, np.newaxis, :]
    A_modified = np.concatenate((dff, tr), axis=1)
    A_modified = A_modified[:, 1:, :]

    return A_modified