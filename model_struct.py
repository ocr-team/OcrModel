from keras import backend as K

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.regularizers import l2
from keras.callbacks import *

rnn_size = 256
l2_rate = 1e-5
width = None
height = 32


input_tensor = Input((width, height, 1))
x = input_tensor
x = Lambda(lambda x:(x-127.0)/127.0)(x)

for j in range(3):
    x = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_uniform', 
               kernel_regularizer=l2(l2_rate))(x)
    x = BatchNormalization(gamma_regularizer=l2(l2_rate), beta_regularizer=l2(l2_rate))(x)
    x = Activation('relu')(x)
x = MaxPooling2D((2, 2))(x)

for j in range(4):
    x = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform', 
               kernel_regularizer=l2(l2_rate))(x)
    x = BatchNormalization(gamma_regularizer=l2(l2_rate), beta_regularizer=l2(l2_rate))(x)
    x = Activation('relu')(x)
x = MaxPooling2D((2, 2))(x)

for j in range(6):
    x = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform', 
               kernel_regularizer=l2(l2_rate))(x)
    x = BatchNormalization(gamma_regularizer=l2(l2_rate), beta_regularizer=l2(l2_rate))(x)
    x = Activation('relu')(x)
x = MaxPooling2D((1, 2))(x)

cnn_model = Model(input_tensor, x, name='cnn')

input_tensor = Input((width, height, 1))
x = cnn_model(input_tensor)

conv_shape = x.get_shape().as_list()
rnn_length = conv_shape[1]
rnn_dimen = conv_shape[3]*conv_shape[2]

print(conv_shape, rnn_length, rnn_dimen)

x = Reshape(target_shape=(-1, rnn_dimen))(x)
rnn_imp = 1

# x = Dense(rnn_size, kernel_initializer='he_uniform', kernel_regularizer=l2(l2_rate))(x)
# x = BatchNormalization(gamma_regularizer=l2(l2_rate), beta_regularizer=l2(l2_rate))(x)
# x = Activation('relu')(x)
x = Dropout(0.2)(x)

for i in range(2):
    x = Conv1D(rnn_size, 3, padding='same', activation='relu', kernel_regularizer=l2(l2_rate))(x)
    x = BatchNormalization(gamma_regularizer=l2(l2_rate), beta_regularizer=l2(l2_rate))(x)
    x = Activation('relu')(x)

x = Dropout(0.2)(x)
x = Dense(n_class, activation='softmax', kernel_regularizer=l2(l2_rate), bias_regularizer=l2(l2_rate))(x)
rnn_out = x
base_model = Model(input_tensor, x)

label_tensor = Input(name='the_labels', shape=[n_len], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')
loss_out = Lambda(ctc_lambda_func, output_shape=(1,), 
                  name='ctc')([x, label_tensor, input_length, label_length])

model = Model(inputs=[input_tensor, label_tensor, input_length, label_length], outputs=[loss_out])
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam')
base_model.save(model_name)
