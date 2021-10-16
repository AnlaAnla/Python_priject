import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential


# 第一个残差模块
class BasicBlock(layers.Layer):
    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = layers.Conv2D(filter_num, (3, 3), strides=stride, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')
        self.conv2 = layers.Conv2D(filter_num, (3, 3), strides=1, padding='same')
        self.bn2 = layers.BatchNormalization()
        # se-block
        self.se_globalpool = keras.layers.GlobalAveragePooling2D()
        self.se_resize = keras.layers.Reshape((1, 1, filter_num))
        self.se_fc1 = keras.layers.Dense(units=filter_num // 16, activation='relu',
                                         use_bias=False)
        self.se_fc2 = keras.layers.Dense(units=filter_num, activation='sigmoid',
                                         use_bias=False)
        if stride != 1:
            self.downsample = Sequential()
            self.downsample.add(layers.Conv2D(filter_num, (1, 1), strides=stride))
        else:
            self.downsample = lambda x: x

    def call(self, input, training=None):
        out = self.conv1(input)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # se_block
        b = out
        out = self.se_globalpool(out)
        out = self.se_resize(out)
        out = self.se_fc1(out)
        out = self.se_fc2(out)
        out = keras.layers.Multiply()([b, out])

        identity = self.downsample(input)
        output = layers.add([out, identity])
        output = tf.nn.relu(output)
        return output


class ResNet(keras.Model):
    def __init__(self, layer_dims, num_classes=10):
        super(ResNet, self).__init__()
        # 预处理层
        self.padding = keras.layers.ZeroPadding2D((3, 3))
        self.stem = Sequential([
            layers.Conv2D(64, (7, 7), strides=(2, 2)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')
        ])
        # resblock
        self.layer1 = self.build_resblock(64, layer_dims[0])
        self.layer2 = self.build_resblock(128, layer_dims[1], stride=2)
        self.layer3 = self.build_resblock(256, layer_dims[2], stride=2)
        self.layer4 = self.build_resblock(512, layer_dims[3], stride=2)
        # 全局池化
        self.avgpool = layers.GlobalAveragePooling2D()
        # 全连接层
        self.fc = layers.Dense(num_classes, activation=tf.keras.activations.softmax)

    def call(self, input, training=None):
        x= self.padding(input)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # [b,c]
        x = self.avgpool(x)
        x = self.fc(x)
        return x

    def build_resblock(self, filter_num, blocks, stride=1):
        res_blocks = Sequential()
        res_blocks.add(BasicBlock(filter_num, stride))
        for pre in range(1, blocks):
            res_blocks.add(BasicBlock(filter_num, stride=1))
        return res_blocks


def ResNet34(num_classes=10):
    return ResNet([2, 2, 2, 2], num_classes=num_classes)


def ResNet34(num_classes=10):
    return ResNet([3, 4, 6, 3], num_classes=num_classes)


model = ResNet34(num_classes=1000)
model.build(input_shape=(1, 224, 224, 3))
print(model.summary())  # 统计网络参数
