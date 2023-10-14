import tensorflow as tf
from keras import backend as K
print(K.backend())
print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
