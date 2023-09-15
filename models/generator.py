from keras.layers import Input,Lambda
from keras.layers.core import Dense
from keras.models import Model

from keras.layers import LSTM,core,concatenate,Bidirectional,Add,GlobalAveragePooling1D
from keras.models import Model,Sequential
from keras import backend as k
from self_attention import Self_Attention

from .networks_utils import (residual_dense_block,residual_bilstm_block,
                             residual_lstm_block,dense_layer)


def resnet_generator_FC_bigger(input_shape=(56,), use_dropout=False, use_batch_norm=True,
                               use_leaky_relu=False):
    inputs = Input(shape=input_shape)
    embedding = inputs

    embedding = dense_layer(embedding, units=56, use_batch_norm=use_batch_norm, use_leaky_relu=use_leaky_relu)

    embedding = residual_dense_block(embedding, units=56, use_dropout=use_dropout, use_batch_norm=use_batch_norm,
                                     use_leaky_relu=use_leaky_relu)
    embedding = residual_dense_block(embedding, units=56, use_dropout=use_dropout, use_batch_norm=use_batch_norm,
                                     use_leaky_relu=use_leaky_relu)
    embedding = residual_dense_block(embedding, units=56, use_dropout=use_dropout, use_batch_norm=use_batch_norm,
                                     use_leaky_relu=use_leaky_relu)
    embedding = residual_dense_block(embedding, units=56, use_dropout=use_dropout, use_batch_norm=use_batch_norm,
                                     use_leaky_relu=use_leaky_relu)

    embedding = dense_layer(embedding, units=56, use_batch_norm=use_batch_norm, use_leaky_relu=use_leaky_relu)

    outputs = Dense(units=input_shape[0], activation=None)(embedding)

    return Model(inputs=inputs, outputs=outputs), inputs, outputs


def resnet_generator_FC_smaller(input_shape=(56,), use_dropout=False, use_batch_norm=True,
                                use_leaky_relu=False):
    inputs = Input(shape=input_shape)
    embedding = inputs

    embedding = residual_dense_block(embedding, units=56, use_dropout=use_dropout, use_batch_norm=use_batch_norm,
                                     use_leaky_relu=use_leaky_relu)
    embedding = residual_dense_block(embedding, units=56, use_dropout=use_dropout, use_batch_norm=use_batch_norm,
                                     use_leaky_relu=use_leaky_relu)
    embedding = residual_dense_block(embedding, units=56, use_dropout=use_dropout, use_batch_norm=use_batch_norm,
                                     use_leaky_relu=use_leaky_relu)
    embedding = residual_dense_block(embedding, units=56, use_dropout=use_dropout, use_batch_norm=use_batch_norm,
                                     use_leaky_relu=use_leaky_relu)

    outputs = Dense(units=input_shape[0], activation=None)(embedding)

    return Model(inputs=inputs, outputs=outputs), inputs, outputs


def resnet_generator_FC_smallest(input_shape=(56,), use_dropout=False, use_batch_norm=True,
                                 use_leaky_relu=False):
    inputs = Input(shape=input_shape)
    embedding = inputs

    embedding = residual_dense_block(embedding, units=56, use_dropout=use_dropout, use_batch_norm=use_batch_norm,
                                     use_leaky_relu=use_leaky_relu)

    outputs = Dense(units=input_shape[0], activation=None)(embedding)

    return Model(inputs=inputs, outputs=outputs), inputs, outputs

def bilstm_att_generator_FC_smallest(input_shape=(56,), use_dropout=False, use_batch_norm=True,
                                 use_leaky_relu=False):
    inputs = Input(shape=input_shape)
    embeddEming0 = inputs

    embedding = core.Lambda(lambda embedding: k.expand_dims(embedding,axis=-1))(embedding0)
    embedding = Bidirectional(LSTM(units=56, return_sequences=True,))(embedding)
    attention_probs = Dense(112, activation='softmax', )(embedding)   #
    embedding = Lambda(lambda x:x[0]*x[1])([attention_probs,embedding])
    #embedding = Self_Attention(112)(embedding)
    embedding = GlobalAveragePooling1D()(embedding)
     
    embedding = concatenate([embedding0,embedding],axis=-1)

    outputs = Dense(units=input_shape[0], activation=None)(embedding)

    return Model(inputs=inputs, outputs=outputs), inputs, outputs


def resbilstm_generator_FC_smaller(input_shape=(56,), use_dropout=False, use_batch_norm=True,
                                 use_leaky_relu=False):
    inputs = Input(shape=input_shape)
    embedding = inputs
    embedding = residual_bilstm_block(embedding, units=56, use_dropout=use_dropout, use_batch_norm=use_batch_norm,
                                     use_leaky_relu=use_leaky_relu)
    embedding = residual_bilstm_block(embedding, units=56, use_dropout=use_dropout, use_batch_norm=use_batch_norm,
                                     use_leaky_relu=use_leaky_relu)
    outputs = Dense(units=input_shape[0], activation=None)(embedding)

    return Model(inputs=inputs, outputs=outputs), inputs, outputs

def resbilstm_generator_FC_smallest(input_shape=(56,), use_dropout=False, use_batch_norm=True,
                                 use_leaky_relu=False):
    inputs = Input(shape=input_shape)
    embedding = inputs
    embedding = residual_bilstm_block(embedding, units=56, use_dropout=use_dropout, use_batch_norm=use_batch_norm,
                                     use_leaky_relu=use_leaky_relu)
    outputs = Dense(units=input_shape[0], activation=None)(embedding)

    return Model(inputs=inputs, outputs=outputs), inputs, outputs

def reslstm_generator_FC_smallest(input_shape=(56,), use_dropout=False, use_batch_norm=True,
                                 use_leaky_relu=False):
    inputs = Input(shape=input_shape)
    embedding = inputs
    embedding = residual_lstm_block(embedding, units=56, use_dropout=use_dropout, use_batch_norm=use_batch_norm,
                                     use_leaky_relu=use_leaky_relu)
    embedding = residual_lstm_block(embedding, units=56, use_dropout=use_dropout, use_batch_norm=use_batch_norm,
                                     use_leaky_relu=use_leaky_relu)
    outputs = Dense(units=input_shape[0], activation=None)(embedding)

    return Model(inputs=inputs, outputs=outputs), inputs, outputs


def resnet_generator(network_type='FC_smaller', **args):
    assert network_type in {'FC_smaller', 'FC_smallest', 'FC_bigger', 'FC_bilstm', 'FC_bilstm_smallest', 'FC_lstm', 'FC_concat', 'FC_bilstm_att'}, "NOT IMPLEMENTED FOR THIS 'network_type'!!!"

    generators = {
        "FC_smaller": resnet_generator_FC_smaller,
        "FC_smallest": resnet_generator_FC_smallest,
        "FC_bigger": resnet_generator_FC_bigger,
        "FC_bilstm": resbilstm_generator_FC_smaller,
        "FC_bilstm_smallest": resbilstm_generator_FC_smallest,
        "FC_lstm": reslstm_generator_FC_smallest,
        "FC_concat":resnet_generator_FC_smallest,
        "FC_bilstm_att":bilstm_att_generator_FC_smallest
    }

    return generators[network_type](**args)
