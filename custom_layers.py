import keras.backend as k
import tensorflow as tf
from keras.engine.topology import Layer
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Concatenate, BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.models import Model

import concerete_dropout as cd
import static_values as sv
from custom_densenet import DenseNet169 as densnet
from custom_vgg19 import VGG19


def get_model(inputs):
    inps = BatchNormalization()(inputs)
    resized_inps = ResizeInp([sv.STATIC_VALUES.image_size[0] / 2, sv.STATIC_VALUES.image_size[1] / 2])(inputs)
    resized_inps = BatchNormalization()(resized_inps)

    with tf.name_scope("xception"):
        x = VGG19(include_top=False, weights='imagenet',
                  input_shape=(sv.STATIC_VALUES.image_size[0], sv.STATIC_VALUES.image_size[1], 3))(resized_inps)

    with tf.name_scope("densnet"):
        y = densnet(include_top=False, weights='imagenet',
                    input_shape=(sv.STATIC_VALUES.image_size[0], sv.STATIC_VALUES.image_size[1], 3))(inps)

    net1 = NetworkInNetwork(10 * sv.STATIC_VALUES.labels_count)(x)
    net2 = NetworkInNetwork(10 * sv.STATIC_VALUES.labels_count)(y)

    slices_net1 = []
    for i in range(0, sv.STATIC_VALUES.labels_count):
        s = Slice(i, 10)(net1)
        s = NetworkInNetwork(10)(s)
        s = NetworkInNetwork(10)(s)
        s = NetworkInNetwork(10)(s)
        s = NetworkInNetwork(10)(s)
        slices_net1.append(s)

    slices_net2 = []
    for i in range(0, sv.STATIC_VALUES.labels_count):
        s = Slice(i, 10)(net2)
        s = NetworkInNetwork(10)(s)
        s = NetworkInNetwork(10)(s)
        s = NetworkInNetwork(10)(s)
        s = NetworkInNetwork(10)(s)
        slices_net2.append(s)

    net1 = Concatenate()(slices_net1)
    net2 = Concatenate()(slices_net2)

    elwise_max = ElementWiseMax()([net1, net2])
    elwise_avg = ElementWiseAvg()([net1, net2])

    max_max = GlobalMaxPooling2D()(elwise_max)
    avg_avg = GlobalAveragePooling2D()(elwise_avg)

    max_avg = GlobalMaxPooling2D()(elwise_avg)
    avg_max = GlobalAveragePooling2D()(elwise_max)

    last = Concatenate()([max_max, avg_avg, max_avg, avg_max])
    z = BatchNormalization()(last)
    with tf.name_scope("fully_connected"):
        z = cd.ConcreteDropout(Dense(512, activation='relu'))(z)
        z = Dense(sv.STATIC_VALUES.labels_count, activation='sigmoid')(z)

    model = Model(inputs=inputs, outputs=z)
    return model


def reshape_z(x):
    res = k.reshape(x, [-1, 3 * 2, 2048])
    return res


def flatten_z(x):
    res = k.reshape(x, [-1, 2048])
    return res


class NetworkInNetwork(Layer):
    def __init__(self, output_dims, **kwargs):
        super(NetworkInNetwork, self).__init__(**kwargs)
        self.trainable = True
        self.output_dims = output_dims

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(1, 1, input_shape[3], int(self.output_dims)),
                                      initializer='glorot_uniform',
                                      trainable=True)
        self.bias = self.add_weight(name='bias',
                                    shape=(int(self.output_dims),),
                                    initializer='glorot_uniform',
                                    trainable=True)

    def call(self, inputs, **kwargs):
        outputs = k.conv2d(inputs, self.kernel)
        outputs = k.bias_add(outputs, self.bias)
        return outputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], self.output_dims)


class Slice(Layer):
    def __init__(self, begin, size, **kwargs):
        super(Slice, self).__init__(**kwargs)
        self.begin = begin * size
        self.size = size

    def build(self, input_shape):
        pass

    def call(self, inputs, **kwargs):
        outputs = tf.slice(inputs, begin=[0, 0, 0, self.begin], size=[-1, -1, -1, self.size])
        return outputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], self.size)


class ElementWiseMax(Layer):
    def __init__(self, **kwargs):
        super(ElementWiseMax, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, inputs, **kwargs):
        input1 = inputs[0]
        input2 = inputs[1]

        expanded_inp1 = tf.expand_dims(input1, axis=1)
        expanded_inp2 = tf.expand_dims(input2, axis=1)

        concatenated = tf.concat([expanded_inp1, expanded_inp2], axis=1)
        outputs = tf.reduce_max(concatenated, axis=1)
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = input_shape[0]
        return (input_shape[0], input_shape[1], input_shape[2], input_shape[3])


class ElementWiseAvg(Layer):
    def __init__(self, **kwargs):
        super(ElementWiseAvg, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, inputs, **kwargs):
        input1 = inputs[0]
        input2 = inputs[1]

        expanded_inp1 = tf.expand_dims(input1, axis=1)
        expanded_inp2 = tf.expand_dims(input2, axis=1)

        concatenated = tf.concat([expanded_inp1, expanded_inp2], axis=1)
        outputs = tf.reduce_mean(concatenated, axis=1)
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = input_shape[0]
        return (input_shape[0], input_shape[1], input_shape[2], input_shape[3])


class ElementWiseAdd(Layer):
    def __init__(self, **kwargs):
        super(ElementWiseAdd, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, inputs, **kwargs):
        input1 = inputs[0]
        input2 = inputs[1]

        expanded_inp1 = tf.expand_dims(input1, axis=1)
        expanded_inp2 = tf.expand_dims(input2, axis=1)

        concatenated = tf.concat([expanded_inp1, expanded_inp2], axis=1)
        outputs = tf.reduce_sum(concatenated, axis=1)
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = input_shape[0]
        return (input_shape[0], input_shape[1], input_shape[2], input_shape[3])


class ElementWiseAvg3(Layer):
    def __init__(self, **kwargs):
        super(ElementWiseAvg3, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, inputs, **kwargs):
        input1 = inputs[0]
        input2 = inputs[1]
        input3 = inputs[2]

        expanded_inp1 = tf.expand_dims(input1, axis=1)
        expanded_inp2 = tf.expand_dims(input2, axis=1)
        expanded_inp3 = tf.expand_dims(input3, axis=1)

        concatenated = tf.concat([expanded_inp1, expanded_inp2, expanded_inp3], axis=1)
        outputs = tf.reduce_mean(concatenated, axis=1)
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = input_shape[0]
        return (input_shape[0], input_shape[1], input_shape[2], input_shape[3])


class ElementWiseMax3(Layer):
    def __init__(self, **kwargs):
        super(ElementWiseMax3, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, inputs, **kwargs):
        input1 = inputs[0]
        input2 = inputs[1]
        input3 = inputs[2]

        expanded_inp1 = tf.expand_dims(input1, axis=1)
        expanded_inp2 = tf.expand_dims(input2, axis=1)
        expanded_inp3 = tf.expand_dims(input3, axis=1)

        concatenated = tf.concat([expanded_inp1, expanded_inp2, expanded_inp3], axis=1)
        outputs = tf.reduce_max(concatenated, axis=1)
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = input_shape[0]
        return (input_shape[0], input_shape[1], input_shape[2], input_shape[3])


class GetHSV(Layer):
    def __init__(self, **kwargs):
        super(GetHSV, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, inputs, **kwargs):
        outputs = tf.image.rgb_to_hsv(inputs)
        return outputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], input_shape[3])


class ResizeInp(Layer):
    def __init__(self, new_size, **kwargs):
        super(ResizeInp, self).__init__(**kwargs)
        self.new_size = new_size

    def build(self, input_shape):
        pass

    def call(self, inputs, **kwargs):
        outputs = tf.image.resize_images(inputs, [int(self.new_size[0]), int(self.new_size[1])])
        return outputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0], int(self.new_size[0]), int(self.new_size[1]), input_shape[3])


class Preprocess(Layer):
    def __init__(self, **kwargs):
        super(Preprocess, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel1 = self.add_weight(name='kernel1',
                                       shape=(5, 5, input_shape[3], 3),
                                       initializer='glorot_uniform',
                                       trainable=True)
        self.bias1 = self.add_weight(name='bias1',
                                     shape=(3,),
                                     initializer='glorot_uniform',
                                     trainable=True)

        self.kernel2 = self.add_weight(name='kernel2',
                                       shape=(3, 3, input_shape[3], 3),
                                       initializer='glorot_uniform',
                                       trainable=True)
        self.bias2 = self.add_weight(name='bias2',
                                     shape=(3,),
                                     initializer='glorot_uniform',
                                     trainable=True)

        self.kernel3 = self.add_weight(name='kernel3',
                                       shape=(1, 1, input_shape[3], 3),
                                       initializer='glorot_uniform',
                                       trainable=True)
        self.bias3 = self.add_weight(name='bias3',
                                     shape=(3,),
                                     initializer='glorot_uniform',
                                     trainable=True)

        self.kernel12 = self.add_weight(name='kernel1',
                                        shape=(5, 5, input_shape[3], 3),
                                        initializer='glorot_uniform',
                                        trainable=True)
        self.bias12 = self.add_weight(name='bias1',
                                      shape=(3,),
                                      initializer='glorot_uniform',
                                      trainable=True)

        self.kernel22 = self.add_weight(name='kernel2',
                                        shape=(3, 3, input_shape[3], 3),
                                        initializer='glorot_uniform',
                                        trainable=True)
        self.bias22 = self.add_weight(name='bias2',
                                      shape=(3,),
                                      initializer='glorot_uniform',
                                      trainable=True)

        self.kernel32 = self.add_weight(name='kernel3',
                                        shape=(1, 1, input_shape[3], 3),
                                        initializer='glorot_uniform',
                                        trainable=True)
        self.bias32 = self.add_weight(name='bias3',
                                      shape=(3,),
                                      initializer='glorot_uniform',
                                      trainable=True)

    def call(self, inputs, **kwargs):
        out1 = k.conv2d(inputs, self.kernel1, padding='same')
        out1 = k.bias_add(out1, self.bias1)

        out2 = k.conv2d(inputs, self.kernel2, padding='same')
        out2 = k.bias_add(out2, self.bias2)

        out3 = k.conv2d(inputs, self.kernel3, padding='same')
        out3 = k.bias_add(out3, self.bias3)

        expanded_out1 = tf.expand_dims(out1, axis=1)
        expanded_out2 = tf.expand_dims(out2, axis=1)
        expanded_out3 = tf.expand_dims(out3, axis=1)

        concatenated = tf.concat([expanded_out1, expanded_out2, expanded_out3], axis=1)
        outputs = tf.reduce_max(concatenated, axis=1)

        out12 = k.conv2d(outputs, self.kernel12, padding='same')
        out12 = k.bias_add(out12, self.bias12)

        out22 = k.conv2d(outputs, self.kernel22, padding='same')
        out22 = k.bias_add(out22, self.bias22)

        out32 = k.conv2d(outputs, self.kernel32, padding='same')
        out32 = k.bias_add(out32, self.bias32)

        expanded_out12 = tf.expand_dims(out12, axis=1)
        expanded_out22 = tf.expand_dims(out22, axis=1)
        expanded_out32 = tf.expand_dims(out32, axis=1)

        concatenated2 = tf.concat([expanded_out12, expanded_out22, expanded_out32], axis=1)
        outputs2 = tf.reduce_max(concatenated2, axis=1)

        return outputs2

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], input_shape[3])


class ApplyConConv(Layer):
    def __init__(self, **kwargs):
        super(ApplyConConv, self).__init__(**kwargs)
        self.trainable = True

    def build(self, input_shape):
        self.kernel1 = self.add_weight(name='kernel',
                                       shape=(1, 1, input_shape[3],
                                              int(input_shape[3] / int(input_shape[1] * input_shape[2]))),
                                       initializer='uniform',
                                       trainable=True)
        self.bias1 = self.add_weight(name='bias',
                                     shape=(int(input_shape[3] / int(input_shape[1] * input_shape[2])),),
                                     initializer='uniform',
                                     trainable=True)

        self.kernel2 = self.add_weight(name='kernel',
                                       shape=(1, 1, int(input_shape[3] / (input_shape[1] * input_shape[2])),
                                              int(input_shape[3] / int(input_shape[1] * input_shape[2]))),
                                       initializer='uniform',
                                       trainable=True)
        self.bias2 = self.add_weight(name='bias',
                                     shape=(int(input_shape[3] / int(input_shape[1] * input_shape[2])),),
                                     initializer='uniform',
                                     trainable=True)

    def call(self, inputs, **kwargs):
        outputs = k.conv2d(inputs, self.kernel1)
        outputs = k.bias_add(outputs, self.bias1)

        outputs = k.conv2d(outputs, self.kernel2)
        outputs = k.bias_add(outputs, self.bias2)

        self.shape = outputs.shape
        outputs = k.reshape(outputs, [-1, self.shape[1] * self.shape[2] * self.shape[3]])
        return outputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[3])


class ConvDeconv(Layer):
    def __init__(self, **kwargs):
        super(ConvDeconv, self).__init__(**kwargs)
        self.trainable = True

    def build(self, input_shape):
        self.conv_k = self.add_weight(name='conv_k', shape=(3, 3, input_shape[3], 256), initializer='uniform',
                                      trainable=True)
        self.conv_b = self.add_weight(name='conv_b', shape=(256,), initializer='uniform', trainable=True)

        self.fc_k = self.add_weight(name='fc_k', shape=(4 * 4 * 256, 256), initializer='uniform', trainable=True)
        self.fc_b = self.add_weight(name='fc_b', shape=(256,), initializer='uniform', trainable=True)

        self.fc_r_k = self.add_weight(name='fc_r_k', shape=(256, 4 * 4 * 256), initializer='uniform', trainable=True)
        self.fc_r_b = self.add_weight(name='fc_r_b', shape=(4 * 4 * 256,), initializer='uniform', trainable=True)

        self.deconv_k = self.add_weight(name='deconv_k', shape=(3, 3, 256, input_shape[3]), initializer='uniform',
                                        trainable=True)
        self.deconv_b = self.add_weight(name='deconv_b', shape=(input_shape[3],), initializer='uniform', trainable=True)

    def call(self, inputs, **kwargs):
        # conv
        outputs = k.conv2d(inputs, self.conv_k)
        outputs = k.bias_add(outputs, self.conv_b)
        outputs = k.pool2d(outputs, (3, 3))
        shape = outputs.shape
        outputs = k.reshape(outputs, [-1, shape[1] * shape[2] * shape[3]])

        # fc
        outputs = k.dot(outputs, self.fc_k)
        outputs = k.bias_add(outputs, self.fc_b)

        outputs = k.dot(outputs, self.fc_r_k)
        outputs = k.bias_add(outputs, self.fc_r_b)

        # deconv
        outputs = k.reshape(outputs, [-1, shape[1], shape[2], shape[3]])
        outputs = UpSampling2D((3, 3))(outputs)
        outputs = k.conv2d(outputs, self.deconv_k)
        outputs = k.bias_add(outputs, self.deconv_b)

        # global max pooling
        outputs = GlobalMaxPooling2D()(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[3])


class Apply_minmax(Layer):
    def __init__(self, **kwargs):
        super(Apply_minmax, self).__init__(**kwargs)
        self.trainable = True

    def build(self, input_shape):
        pass

    def call(self, inputs, **kwargs):
        min = k.min(inputs, axis=[1, 2])
        max = k.max(inputs, axis=[1, 2])
        outputs = max - min
        return outputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[3])
