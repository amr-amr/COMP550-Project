from keras import backend as K
from keras.layers import Input, Lambda
from keras.models import Model
import tensorflow as tf
from keras.engine.topology import Layer
import numpy as np

a = tf.ones((4, 2))
b = tf.ones((2, 5))
c = tf.matmul(a, b)

# a = K.ones((4, 2))
# b = K.ones((2, 5))
# c = K.batch_dot(a, b)


class MyLayer(Layer):

    def __init__(self, output_shape, **kwargs):
        self._output_shape = output_shape
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(MyLayer, self).build(input_shape)

    def call(self, x):
        return K.batch_dot(x[0], x[1])

    def compute_output_shape(self, input_shape):
        return self._output_shape



sent_dim = 2
wv_dim = 5
wv_input = Input(shape=(sent_dim, wv_dim), name='wv_input')
filter_input = Input(shape=(sent_dim, sent_dim), name='filter_input')


# a = K.ones((4, 2))
# b = K.ones((2, 5))
# c = K.dot(filter_input, wv_input)

# mult = MyLayer(output_shape=(sent_dim, wv_dim))([filter_input, wv_input])

def stupid_function(x):
    return K.batch_dot(x[0], x[1])


mult = Lambda(lambda x: K.batch_dot(x[0], x[1]), output_shape=(sent_dim, wv_dim))([filter_input, wv_input])

# mult = Lambda(K.batch_dot, output_shape=(sent_dim, wv_dim))(filter_input, wv_input)


model = Model(inputs=[filter_input, wv_input], outputs=mult)

filt = np.zeros((2, 2, 2))
filt[:][1][1] = -1
# filt[1][1] = -1
# filt[0][1] = -1*filt[0][1]
# filt[1][1] = -1*filt[1][1]
wv = np.ones((2, 2, 5))
out = model.predict([filt, wv])
print(out)