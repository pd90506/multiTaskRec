import tensorflow.keras as keras
import tensorflow.keras.backend as K 
import numpy as np
import tensorflow as tf
tf.enable_eager_execution()

def hr_metric(y_true, y_pred, k=10):
    hits = K.equal(y_true, y_pred)
    hits = K.cast(hits, 'int32')
    return K.sum(hits)



if __name__ == '__main__':
    a = np.random.randint(2, size=100)
    b = np.random.randint(2, size=100)
    hr = hr_metric(a, b)
    print(hr)