import tensorflow as tf
import tensorflow.keras.layers as layers


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = tf.keras.Sequential([
            layers.Conv2D(outchannel, kernel_size=3, strides=stride, padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(outchannel, kernel_size=3, strides=1, padding='same', use_bias=False),
        ], name='left')

        self.shortcut = tf.keras.Sequential(name='shortcut')
        if stride != 1 or inchannel != outchannel:
            self.shortcut.add(layers.Conv2D(outchannel, kernel_size=1, strides=stride, use_bias=False))
            self.shortcut.add(layers.BatchNormalization())

    def call(self, inputs, training=None):
        out = self.left(inputs)
        out += self.shortcut(inputs)
        out = tf.nn.relu(out)
        return out


class ResNet(tf.keras.Model):
    def __init__(self, num_classes=8):
        super(ResNet, self).__init__()
        self.inchannel = 16
        self.conv_0 = tf.keras.Sequential([
            layers.Conv2D(16, kernel_size=3, strides=1, padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU()
        ], name='conv_0')
        self.res_1 = tf.keras.Sequential(self.make_layer(16, 3, 1), name='res_1')
        self.res_2 = tf.keras.Sequential(self.make_layer(32, 3, 2), name='res_2')
        self.res_3 = tf.keras.Sequential(self.make_layer(64, 3, 2), name='res_3')
        self.fc_4 = tf.keras.Sequential([
            layers.Reshape([64]),
            layers.Dense(num_classes)
        ], name='fc_4')

    def make_layer(self, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(self.inchannel, channels, stride))
            self.inchannel = channels
        return layers

    def call(self, inputs, training=None, mask=None):
        out = self.conv_0(inputs)
        out = self.res_1(out)
        out = self.res_2(out)
        out = self.res_3(out)
        out = tf.nn.avg_pool2d(out, out.shape[1], 1, padding='VALID')
        out = self.fc_4(out)
        return out


class LogisticRegression(tf.keras.models.Model):
    def __init__(self, num_classes):
        super(LogisticRegression, self).__init__()
        self.fc_0 = tf.keras.Sequential([
            layers.Dense(num_classes, activation=tf.keras.activations.sigmoid, dtype='float64')
        ], name='fc_0')

    def call(self, inputs, training=None, mask=None):
        out = self.fc_0(inputs)
        return out


class ConvolutionalNetwork(tf.keras.models.Model):
    def __init__(self, num_classes=10):
        super(ConvolutionalNetwork, self).__init__()
        self.cnn_0 = tf.keras.Sequential([
            layers.Conv2D(30, 3, 1, padding='same'),
            layers.BatchNormalization(trainable=False),
            layers.ReLU()
        ], name='cnn_0')
        self.max_pool_0 = tf.keras.Sequential([
            layers.MaxPool2D((2, 2), 2, padding='VALID')
        ], name='max_pool_0')
        self.cnn_1 = tf.keras.Sequential([
            layers.Conv2D(50, 3, 1, padding='same'),
            layers.BatchNormalization(trainable=False),
            layers.ReLU()
        ], name='cnn_1')
        self.max_pool_1 = tf.keras.Sequential([
            layers.MaxPool2D((2, 2), 2, padding='VALID')
        ], name='max_pool_1')
        self.fc_2 = tf.keras.Sequential([
            layers.Reshape([50]),
            layers.Dense(num_classes)
        ], name='fc_2')

    def call(self, inputs, training=None, mask=None):
        out = self.cnn_0(inputs)
        out = self.max_pool_0(out)
        out = self.cnn_1(out)
        out = self.max_pool_1(out)
        out = tf.nn.avg_pool2d(out, out.shape[1], strides=1, padding='VALID')
        out = self.fc_2(out)
        return out


class Generator_CNN(tf.keras.models.Model):
    def __init__(self):
        super(Generator_CNN, self).__init__()
        self.fc_0 = tf.keras.Sequential([
            layers.Dense(7 * 7 * 256, use_bias=False),
            layers.BatchNormalization(trainable=False),
            layers.LeakyReLU(),
            layers.Reshape((7, 7, 256))
        ], name='fc_0')
        self.convt_1 = tf.keras.Sequential([
            layers.Conv2DTranspose(128, 5, 1, padding='same', use_bias=False),
            layers.BatchNormalization(trainable=False),
            layers.LeakyReLU()
        ], name='convt_1')
        self.convt_2 = tf.keras.Sequential([
            layers.Conv2DTranspose(64, 5, 2, padding='same', use_bias=False),
            layers.BatchNormalization(trainable=False),
            layers.LeakyReLU()
        ], name='convt_2')
        self.convt_3 = tf.keras.Sequential([
            layers.Conv2DTranspose(1, 5, 2, padding='same', use_bias=False)
        ], name='convt_3')

    def call(self, inputs, training=None, mask=None):
        out = self.fc_0(inputs)
        out = self.convt_1(out)
        out = self.convt_2(out)
        out = self.convt_3(out)
        return out


class Discriminator_CNN(tf.keras.models.Model):
    def __init__(self):
        super(Discriminator_CNN, self).__init__()
        self.conv_0 = tf.keras.Sequential([
            layers.Conv2D(64, 5, 2, padding='same'),
            layers.LeakyReLU(),
            layers.Dropout(0.3)
        ], name='conv_0')
        self.conv_1 = tf.keras.Sequential([
            layers.Conv2D(128, 5, 2, padding='same'),
            layers.LeakyReLU(),
            layers.Dropout(0.3)
        ], name='conv_1')
        self.fc_2 = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(1)
        ], name='fc_2')

    def call(self, inputs, training=None, mask=None):
        out = self.conv_0(inputs)
        out = self.conv_1(out)
        out = self.fc_2(out)
        return out


class Generator_LR(tf.keras.models.Model):
    def __init__(self, out_shape):
        super(Generator_LR, self).__init__()
        self.fc_0 = tf.keras.Sequential([
            layers.Dense(1024, use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU()
        ], name='fc_0')
        self.fc_1 = tf.keras.Sequential([
            layers.Dense(out_shape, use_bias=False)
        ], name='fc_1')

    def call(self, inputs, training=None, mask=None):
        out = self.fc_0(inputs)
        out = self.fc_1(out)
        return out


class Discriminator_LR(tf.keras.models.Model):
    def __init__(self):
        super(Discriminator_LR, self).__init__()
        self.fc_0 = tf.keras.Sequential([
            layers.Dense(1, activation=tf.keras.activations.sigmoid)
        ], name='fc_0')

    def call(self, inputs, training=None, mask=None):
        out = self.fc_0(inputs)
        return out


class Generator_RES(tf.keras.models.Model):
    def __init__(self, *out_shape):
        super(Generator_RES, self).__init__()
        size = int(out_shape[1] / 2 / 2)
        self.fc_0 = tf.keras.Sequential([
            layers.Dense(size * size * 256, use_bias=False),
            layers.BatchNormalization(trainable=False),
            layers.LeakyReLU(),
            layers.Reshape((size, size, 256))
        ], name='fc_0')
        self.convt_1 = tf.keras.Sequential([
            layers.Conv2DTranspose(128, 5, 1, padding='same', use_bias=False),
            layers.BatchNormalization(trainable=False),
            layers.LeakyReLU()
        ], name='convt_1')
        self.convt_2 = tf.keras.Sequential([
            layers.Conv2DTranspose(64, 5, 2, padding='same', use_bias=False),
            layers.BatchNormalization(trainable=False),
            layers.LeakyReLU()
        ], name='convt_2')
        self.convt_3 = tf.keras.Sequential([
            layers.Conv2DTranspose(3, 5, 2, padding='same', use_bias=False)
        ], name='convt_3')

    def call(self, inputs, training=None, mask=None):
        out = self.fc_0(inputs)
        out = self.convt_1(out)
        out = self.convt_2(out)
        out = self.convt_3(out)
        return out


class Discriminator_RES(tf.keras.models.Model):
    def __init__(self):
        super(Discriminator_RES, self).__init__()
        self.conv_0 = tf.keras.Sequential([
            layers.Conv2D(64, 5, 2, padding='same'),
            layers.LeakyReLU(),
            layers.Dropout(0.3)
        ], name='conv_0')
        self.conv_1 = tf.keras.Sequential([
            layers.Conv2D(128, 5, 2, padding='same'),
            layers.LeakyReLU(),
            layers.Dropout(0.3)
        ], name='conv_1')
        self.fc_2 = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(1)
        ], name='fc_2')

    def call(self, inputs, training=None, mask=None):
        out = self.conv_0(inputs)
        out = self.conv_1(out)
        out = self.fc_2(out)
        return out
