from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from ..Utils.data_evaluation import data_evaluation

class RecurrentNeuralNetworkPipeline:
    def __init__(self, X_train, X_test,
                 y_train, y_test,
                 args, parameters=None):
        self.args = args
        self.parameters = parameters or {}
        self.model = None
        self.results = None

        # Store pre‑prepared splits directly
        self.X_train = X_train
        self.X_test  = X_test
        self.y_train = y_train
        self.y_test  = y_test

    def build_model(self):
        # Extract parameters with defaults
        units = self.parameters.get("units", 64)
        activation = self.parameters.get("activation", "tanh")
        optimizer = self.parameters.get("optimizer", "adam")
        loss = self.parameters.get("loss", "binary_crossentropy")
        metrics = self.parameters.get("metrics", ["accuracy"])
        input_shape = self.parameters.get("input_shape")

        if input_shape is None:
            raise ValueError("You must provide 'input_shape' in parameters for RNN.")

        # Build a simple RNN model
        self.model = Sequential()
        self.model.add(SimpleRNN(units, activation=activation, input_shape=input_shape))
        self.model.add(Dense(1, activation="sigmoid"))  # Binary classification output

        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return self

    def train(self, epochs=10, batch_size=32):
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        self.model.fit(self.X_train, self.y_train,
                       epochs=epochs,
                       batch_size=batch_size,
                       validation_data=(self.X_test, self.y_test),
                       verbose=1)
        return self

    def evaluation(self):
        self.results = data_evaluation(self.model, self.X_test, self.y_test)
        return self

    def show_results(self):
        return self.results