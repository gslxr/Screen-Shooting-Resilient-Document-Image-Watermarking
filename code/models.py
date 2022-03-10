import tensorflow as tf
import utils
from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *


# Encoder
class wtrmarkEncoder(Layer):
    def __init__(self):
        super(wtrmarkEncoder, self).__init__()

        self.secret_dense = Dense(7500, activation='relu', kernel_initializer='he_normal')
        self.conv1 = Sequential([
                        Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal'),
                        Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')])
        self.conv2 = Sequential([
                        Conv2D(32, 3, activation='relu', strides=2, padding='same', kernel_initializer='he_normal'),
                        Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')])
        self.conv3 = Sequential([
                        Conv2D(64, 3, activation='relu', strides=2, padding='same', kernel_initializer='he_normal'),
                        Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')])
        self.conv4 = Sequential([
                        Conv2D(128, 3, activation='relu', strides=2, padding='same', kernel_initializer='he_normal'),
                        Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')])
        self.conv5 = Sequential([
                        Conv2D(256, 3, activation='relu', strides=2, padding='same', kernel_initializer='he_normal'),
                        Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')])
        self.up6 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv6 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.up7 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv7 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.up8 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv8 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.up9 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv9 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv10 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.residual = Conv2D(3, 1, activation=None, padding='same', kernel_initializer='he_normal')

    def call(self, inputs):
        secret, image = inputs

        secret = self.secret_dense(secret)
        secret = Reshape((50, 50, 3))(secret)
        secret_enlarged = UpSampling2D(size=(8, 8))(secret)

        inputs = concatenate([secret_enlarged, image], axis=-1)
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        up6 = self.up6(UpSampling2D(size=(2, 2))(conv5))
        merge6 = concatenate([conv4, up6], axis=3)
        conv6 = self.conv6(merge6)
        up7 = self.up7(UpSampling2D(size=(2, 2))(conv6))
        merge7 = concatenate([conv3, up7], axis=3)
        conv7 = self.conv7(merge7)
        up8 = self.up8(UpSampling2D(size=(2, 2))(conv7))
        merge8 = concatenate([conv2, up8], axis=3)
        conv8 = self.conv8(merge8)
        up9 = self.up9(UpSampling2D(size=(2, 2))(conv8))
        merge9 = concatenate([conv1, up9], axis=3)
        conv9 = self.conv9(merge9)
        conv10 = self.conv10(conv9)
        residual = self.residual(conv10)
        return residual


# Decoder
class wtrmarkDecoder(Layer):
    def __init__(self, secret_size):
        super(wtrmarkDecoder, self).__init__()

        self.decoder = Sequential([
            Conv2D(16, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal'),
            Conv2D(16, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal'),
            Conv2D(32, (3, 3), strides=2, activation='relu', padding='same',kernel_initializer='he_normal'),
            Conv2D(32, (3, 3), strides=2, activation='relu', padding='same',kernel_initializer='he_normal'),
            Conv2D(32, (3, 3), strides=2, activation='relu', padding='same',kernel_initializer='he_normal'),
            Conv2D(64, (3, 3), strides=2, activation='relu', padding='same',kernel_initializer='he_normal'),
            Conv2D(128, (3, 3), strides=2, activation='relu', padding='same',kernel_initializer='he_normal'),
            Flatten(),
            Dense(secret_size)
        ])

    def call(self, image):
        decoded_secret = self.decoder(image)
        return decoded_secret


# make distortion layer
def distortion_layer(encoded_image, global_step):
    ramp_fn = lambda ramp: tf.minimum(tf.to_float(global_step) / ramp, 1.)  # (?/1000,1.0) = (0,1.0) ?:[0,100000]

    # Gaussian Blur2
    filter_blur = utils.random_blur_kernel(probs=[.25, .25], N_blur=7, sigrange_gauss=[1., 3.], sigrange_line=[.25, 1.], wmin_line=3)
    encoded_image = tf.nn.conv2d(encoded_image, filter_blur, [1, 1, 1, 1], padding='SAME')

    # Noise 0.02
    rnd_noise = 0.02
    rnd_noise_ramp = 1000
    rnd_noise = tf.random.uniform([]) * ramp_fn(rnd_noise_ramp) * rnd_noise
    random_noise = tf.random_normal(shape=tf.shape(encoded_image), mean=0.0, stddev=rnd_noise, dtype=tf.float32)
    encoded_image = encoded_image + random_noise
    encoded_image = tf.clip_by_value(encoded_image, 0, 1)

    # Contrast
    contrast_low = 0.5
    contrast_high = 1.5
    contrast_ramp = 1000
    contrast_low = 1. - (1. - contrast_low) * ramp_fn(contrast_ramp)
    contrast_high = 1. + (contrast_high - 1.) * ramp_fn(contrast_ramp)
    contrast_params = [contrast_low, contrast_high]
    contrast_scale = tf.random_uniform(shape=[tf.shape(encoded_image)[0]], minval=contrast_params[0], maxval=contrast_params[1])
    contrast_scale = tf.reshape(contrast_scale, shape=[tf.shape(encoded_image)[0], 1, 1, 1])

    # Brightness
    rnd_bri = 0.3
    batch_size = 4
    rnd_bri_ramp = 1000
    rnd_bri = ramp_fn(rnd_bri_ramp) * rnd_bri
    rnd_brightness = tf.random.uniform((batch_size,1,1,1), -rnd_bri, rnd_bri)

    # Brightness & Contrast
    encoded_image = encoded_image * contrast_scale + rnd_brightness

    # Hue
    rnd_hue = 0.1
    rnd_hue_ramp = 1000
    rnd_hue = ramp_fn(rnd_hue_ramp) * rnd_hue
    rnd_hue = tf.random.uniform((batch_size,1,1,3), -rnd_hue, rnd_hue)
    encoded_image = encoded_image + rnd_hue
    encoded_image = tf.clip_by_value(encoded_image, 0, 1)

    # saturation [1.0]
    rnd_sat = 1.0
    rnd_sat_ramp = 1000
    rnd_sat = tf.random.uniform([]) * ramp_fn(rnd_sat_ramp) * rnd_sat
    encoded_image_lum = tf.expand_dims(tf.reduce_sum(encoded_image * tf.constant([.3, .6, .1]), axis=3), 3)
    encoded_image = (1 - rnd_sat) * encoded_image + rnd_sat * encoded_image_lum
    encoded_image = tf.reshape(encoded_image, [-1, 400, 400, 3])

    # Jpeg Compression 
    jpeg_quality = 50
    jpeg_quality_ramp = 1000
    jpeg_quality = 100. - tf.random.uniform([]) * ramp_fn(jpeg_quality_ramp) * (100. - jpeg_quality)
    jpeg_factor = tf.cond(tf.less(jpeg_quality, 50), lambda: 5000. / jpeg_quality, lambda: 200. - jpeg_quality * 2) / 100. + .0001
    encoded_image = utils.jpeg_compress_decompress(encoded_image, rounding=utils.round_only_at_0, factor=jpeg_factor, downsample_c=True)
    
    return encoded_image


# build model
def build_model(encoder,
                decoder,
                secret_input,
                image_input,
                factor,
                M,
                loss_scales,
                yuv_scales,
                rgb_scales,
                global_step):

    # encoder
    residual = encoder((secret_input, image_input))
    """
      Embedding strength adjustment strategy is meant to change the factor to get high visual quality of the document image.
      you can set the factor value before the mode training, or the default value is 1.0, 
      the more detail you can read the paper from the link: 
      
    """
    encoded_image = image_input + factor * residual
    # encoded_warped
    encoded_warped = tf.contrib.image.transform(encoded_image, M[:, 1, :], interpolation='BILINEAR')
    # transformed_unwarped
    transformed_unwarped = tf.contrib.image.transform(encoded_warped, M[:, 0, :], interpolation='BILINEAR')
    # encoded_unwarped
    encoded_unwarped = distortion_layer(transformed_unwarped, global_step)
    # decoder
    decoded_secret = decoder(encoded_unwarped)

    encoded_image_yuv = tf.image.rgb_to_yuv(encoded_image)
    image_input_yuv = tf.image.rgb_to_yuv(image_input)
    im_yuv_diff = tf.abs(encoded_image_yuv - image_input_yuv)
    yuv_loss_op = tf.reduce_mean(tf.square(im_yuv_diff), axis=[0, 1, 2])  # MSE
    # image_loss
    image_loss_op = tf.tensordot(yuv_loss_op, yuv_scales, axes=1)

    # secret_loss
    secret_loss_op = tf.losses.sigmoid_cross_entropy(secret_input, decoded_secret)

    im_rgb_diff = tf.abs(encoded_image - image_input)
    text_diff_op = im_rgb_diff * tf.convert_to_tensor(1.0 - image_input, dtype=tf.float32)
    diff_loss_op = tf.reduce_mean(text_diff_op, axis=[0, 1, 2])
    # text_loss
    text_loss_op = tf.tensordot(diff_loss_op, rgb_scales, axes=1)

    loss_op = loss_scales[0] * image_loss_op + loss_scales[1] * secret_loss_op + loss_scales[2] * text_loss_op

    return loss_op, image_loss_op, secret_loss_op, text_loss_op


# prepare deployment hiding graph
def prepare_deployment_hiding_graph(encoder, secret_input, image_input):
    residual = encoder((secret_input, image_input))
    encoded_image = image_input + residual
    encoded_image = tf.clip_by_value(encoded_image, 0, 1)

    return encoded_image


# preapre deployemnt reveal graph
def prepare_deployment_reveal_graph(decoder, image_input):
    decoded_secret = decoder(image_input)

    return tf.round(tf.sigmoid(decoded_secret))
