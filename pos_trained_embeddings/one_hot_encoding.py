# https://gist.github.com/bzamecnik/a33052ec46ee7efeb217856d98a4fb5f
"""
When traing ML models on text we usually need to represent words/character in one-hot encoding.
This can be done in preprocessing, however it may make the dataset file bigger. Also when we'd
like to use an Embedding layer, it accepts the original integer indexes instead of one-hot codes.
Can be move the one-hot encoding from pre-preprocessing directly into the model?
If so we could choose from two options: use one-hot inputs or perform embedding.
A way how to do this was suggested in Keras issue [#3680](https://github.com/fchollet/keras/issues/3680).
Actually there's not need to implement a separate layer. We only need to apply
K.one_hot backend operation in the Lambda layer.
There are a few catches when using Lambda(K.one_hot):
- the input must be integer (uint8, int32, int64), not float32
- you have to specify the number of classes explicitly
- you have to specify the output shape explicitly
Here's a working example of how to accomplish this.
Tested in Keras 1.1.2, TensorFlow 0.12.0rc1 Mac OSX CPU.
See also Demultiplexing inputs within Keras layers:
https://gist.github.com/bzamecnik/a4bf0d70ea86c54617aa06aaf6e41615
"""

import numpy as np
from keras import backend as K
from keras.layers import Input, Lambda
from keras.models import Model

# 5 sequences of length 10
nb_sequences = 5
seq_length = 10
nb_classes = 20

input_shape = (seq_length,)
output_shape = (input_shape[0], nb_classes)

# uint8 is ok for <= 256 classes, otherwise use int32
input = Input(shape=input_shape, dtype='uint8')

# Without the output_shape, Keras tries to infer it using calling the function
# on an float32 input, which results in error in TensorFlow:
#
#   TypeError: DataType float32 for attr 'TI' not in list of allowed values: uint8, int32, int64

x_ohe = Lambda(K.one_hot,
               arguments={'num_classes': nb_classes},
               output_shape=output_shape)(input)

x_classes = np.random.randint(0, nb_classes, size=(nb_sequences, seq_length))

out = Model(input, x_ohe).predict(x_classes)

assert x_classes.shape == (nb_sequences, seq_length)
assert Model(input, x_ohe).predict(x_classes).shape == (nb_sequences, seq_length, nb_classes)