import tensorflow as tf

def build_generator():
    inputs = tf.keras.layers.Input(shape=[256, 256, 3])

    down1 = tf.keras.layers.Conv2D(64, 4, strides=2, padding='same', use_bias=False)(inputs)
    down1 = tf.keras.layers.LeakyReLU()(down1)
    down2 = tf.keras.layers.Conv2D(128, 4, strides=2, padding='same', use_bias=False)(down1)
    down2 = tf.keras.layers.BatchNormalization()(down2)
    down2 = tf.keras.layers.LeakyReLU()(down2)

    bottleneck = tf.keras.layers.Conv2D(256, 4, strides=2, padding='same', use_bias=False)(down2)
    bottleneck = tf.keras.layers.BatchNormalization()(bottleneck)
    bottleneck = tf.keras.layers.LeakyReLU()(bottleneck)

    up1 = tf.keras.layers.Conv2DTranspose(128, 4, strides=2, padding='same', use_bias=False)(bottleneck)
    up1 = tf.keras.layers.BatchNormalization()(up1)
    up1 = tf.keras.layers.ReLU()(up1)
    up1 = tf.keras.layers.Concatenate()([up1, down2])

    up2 = tf.keras.layers.Conv2DTranspose(64, 4, strides=2, padding='same', use_bias=False)(up1)
    up2 = tf.keras.layers.BatchNormalization()(up2)
    up2 = tf.keras.layers.ReLU()(up2)
    up2 = tf.keras.layers.Concatenate()([up2, down1])

    outputs = tf.keras.layers.Conv2D(3, 4, strides=1, padding='same', activation='tanh')(up2)

    return tf.keras.models.Model(inputs=inputs, outputs=outputs)

def build_discriminator():
    inputs = tf.keras.layers.Input(shape=[256, 256, 3])
    targets = tf.keras.layers.Input(shape=[256, 256, 3])
    x = tf.keras.layers.Concatenate()([inputs, targets])

    x = tf.keras.layers.Conv2D(64, 4, strides=2, padding='same')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(128, 4, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(256, 4, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(512, 4, strides=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(1, 4, strides=1, padding='same')(x)

    return tf.keras.models.Model(inputs=[inputs, targets], outputs=x)
