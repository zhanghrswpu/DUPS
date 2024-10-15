from model import *


class Client():
    def __init__(self, X_train, y_train):
        self.datasets = {
            'x': X_train,
            'y': y_train
        }
        self.model = None
        self.grads = None

    def client_train(self, model_weights, model_name, classes_num, *input_shape):
        X_train = self.datasets['x']
        y_train = self.datasets['y']
        # Initialize the client model
        self.model_init(model_name, classes_num, model_weights, *input_shape)
        # training model
        self.model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9),
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
        self.model.fit(X_train, y_train, epochs=1, verbose=0, batch_size=32)

        new_model_weights = self.model.get_weights()
        temp_grads = []
        for i in range(len(new_model_weights)):
            temp_grads.append(new_model_weights[i] - model_weights[i])
        self.grads = temp_grads
        self.model = None
        return self

    def model_init(self, model_name, classes_num, server_model_weights, *input_shape):
        if model_name == 'CNN':
            self.model = ConvolutionalNetwork()
        elif model_name == 'LR':
            self.model = LogisticRegression(classes_num)
        else:
            self.model = ResNet(classes_num)
        self.model.build(input_shape=input_shape)
        self.model.set_weights(server_model_weights)

    def get_grads(self):
        return self.grads
