from keras.layers import Input,Lambda
from keras.layers.core import Dense
from keras.models import Model

from .networks_utils import dense_layer, concat_dense_block


def n_layer_discriminator_FC_bigger(input_shape=(56,), use_wgan=False, use_batch_norm=False, use_leaky_relu=False):
    inputs = Input(shape=input_shape)
    x = inputs

    x = dense_layer(x, units=42, use_batch_norm=use_batch_norm, use_leaky_relu=use_leaky_relu)
    x = dense_layer(x, units=42, use_batch_norm=use_batch_norm, use_leaky_relu=use_leaky_relu)

    x = dense_layer(x, units=28, use_batch_norm=use_batch_norm, use_leaky_relu=use_leaky_relu)
    x = dense_layer(x, units=28, use_batch_norm=use_batch_norm, use_leaky_relu=use_leaky_relu)

    x = dense_layer(x, units=14, use_batch_norm=use_batch_norm, use_leaky_relu=use_leaky_relu)
    x = dense_layer(x, units=14, use_batch_norm=use_batch_norm, use_leaky_relu=use_leaky_relu)

    x = dense_layer(x, units=7, use_batch_norm=use_batch_norm, use_leaky_relu=use_leaky_relu)
    x = dense_layer(x, units=7, use_batch_norm=use_batch_norm, use_leaky_relu=use_leaky_relu)

    activation = None if use_wgan else "sigmoid"
    x = Dense(units=1, activation=activation)(x)

    outputs = x
    return Model(inputs=inputs, outputs=outputs)


def n_layer_discriminator_FC_smaller(input_shape=(56,), use_wgan=False, use_batch_norm=False, use_leaky_relu=False):
    inputs = Input(shape=input_shape)
    x = inputs

    x = dense_layer(x, units=48, use_batch_norm=use_batch_norm, use_leaky_relu=use_leaky_relu)
    x = dense_layer(x, units=36, use_batch_norm=use_batch_norm, use_leaky_relu=use_leaky_relu)
    x = dense_layer(x, units=28, use_batch_norm=use_batch_norm, use_leaky_relu=use_leaky_relu)
    x = dense_layer(x, units=18, use_batch_norm=use_batch_norm, use_leaky_relu=use_leaky_relu)
    x = dense_layer(x, units=12, use_batch_norm=use_batch_norm, use_leaky_relu=use_leaky_relu)
    x = dense_layer(x, units=7, use_batch_norm=use_batch_norm, use_leaky_relu=use_leaky_relu)

    activation = None if use_wgan else "sigmoid"
    x = Dense(units=1, activation=activation)(x)

    outputs = x
    return Model(inputs=inputs, outputs=outputs)


def n_layer_discriminator_FC_smallest(input_shape=(56,), use_wgan=False, use_batch_norm=False, use_leaky_relu=False):
    inputs = Input(shape=input_shape)
    x = inputs

    x = dense_layer(x, units=56, use_batch_norm=use_batch_norm, use_leaky_relu=use_leaky_relu)
    x = dense_layer(x, units=28, use_batch_norm=use_batch_norm, use_leaky_relu=use_leaky_relu)
    x = dense_layer(x, units=7, use_batch_norm=use_batch_norm, use_leaky_relu=use_leaky_relu)

    activation = None if use_wgan else "sigmoid"
    x = Dense(units=1, activation=activation)(x)

    outputs = x
    return Model(inputs=inputs, outputs=outputs)

def n_layer_discriminator_FC_smallest_concat(input_shape=(56,), use_wgan=False, use_batch_norm=False, use_leaky_relu=False):
    inputs = Input(shape=input_shape)
    x = inputs

    x = concat_dense_block(x, units=56, use_batch_norm=use_batch_norm, use_leaky_relu=use_leaky_relu)
    x = concat_dense_block(x, units=28, use_batch_norm=use_batch_norm, use_leaky_relu=use_leaky_relu)
    x = concat_dense_block(x, units=7, use_batch_norm=use_batch_norm, use_leaky_relu=use_leaky_relu)

    activation = None if use_wgan else "sigmoid"
    x = Dense(units=1, activation=activation)(x)

    outputs = x
    return Model(inputs=inputs, outputs=outputs)

def n_layer_discriminator_FC_smallest_att(input_shape=(56,), use_wgan=False, use_batch_norm=False, use_leaky_relu=False):
    inputs = Input(shape=input_shape)
    x0 = inputs

    x = dense_layer(x0, units=56, use_batch_norm=use_batch_norm, use_leaky_relu=use_leaky_relu)
    attention_probs = Dense(56, activation='softmax', )(x)   #[b_size,maxlen,64]
    x = Lambda(lambda x:x[0]*x[1])([attention_probs,x])
    x = dense_layer(x, units=28, use_batch_norm=use_batch_norm, use_leaky_relu=use_leaky_relu)
    attention_probs = Dense(28, activation='softmax', )(x)   #[b_size,maxlen,64]
    x = Lambda(lambda x:x[0]*x[1])([attention_probs,x])
    x = dense_layer(x, units=14, use_batch_norm=use_batch_norm, use_leaky_relu=use_leaky_relu)

    activation = None if use_wgan else "sigmoid"
    x = Dense(units=1, activation=activation)(x)

    outputs = x
    return Model(inputs=inputs, outputs=outputs)


def n_layer_discriminator(network_type='FC_smaller', **args):
    assert network_type in {'FC_smaller', 'FC_smallest', 'FC_bigger', 'FC_bilstm', 'FC_bilstm_smallest', 'FC_lstm', 'FC_concat', 'FC_bilstm_att'}, "NOT IMPLEMENTED FOR THIS 'network_type'!!!"

    generators = {
        "FC_smaller": n_layer_discriminator_FC_smaller,
        "FC_smallest": n_layer_discriminator_FC_smallest,
        "FC_bigger": n_layer_discriminator_FC_bigger,
        "FC_bilstm": n_layer_discriminator_FC_smallest_att,
        "FC_lstm": n_layer_discriminator_FC_smallest_att,
        "FC_concat":n_layer_discriminator_FC_smallest_concat,
        "FC_bilstm_att": n_layer_discriminator_FC_smallest_att,
        "FC_bilstm_smallest": n_layer_discriminator_FC_smallest_att,
    }

    return generators[network_type](**args)
