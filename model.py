import numpy as np
import tensorflow as tf
import toml

from data import *
from typing import *
from tensorflow.keras import utils
from tensorflow.keras import Model as KerasModel
from keras.models import Sequential
from keras.layers import Dense, TextVectorization, Dropout, concatenate, Input, Embedding, \
                         Bidirectional, LSTM, GRU, MaxPool1D, Flatten, Conv1D
from keras.constraints import MaxNorm
from keras.callbacks import EarlyStopping, ModelCheckpoint


# read config file first and set constants
loaded = toml.load(os.path.join(os.getcwd(), 'config.toml'))
MAX = loaded['globals']['GLOBAL_MAX']
MAX_TOKENS = loaded['globals']['MAX_TOKENS']
MAX_PADDING = loaded['globals']['MAX_PADDING']


class Model:
    """This Class contains methods used for training a standalone or ensemble model to use to classify tokens"""

    def __init__(self, data: ModelData, ensemble: bool = True, ensemble_count: Optional[int] = None):
        """
        Initialises the MLP Model Class

        Parameters
        ----------
        data:               ModelData instance containing all the data to use for training
        ensemble:           Flag to indicate to create an emsemble model or not
        ensemble_count:     Number of ensemble models to instantiate and train; this parameter means nothing
                            if ensemble training is not done, and a warning will be returned
        """

        self.model = None
        self.file_counter = 0
        self.history = None
        self.ensemble_model = None
        self.ensemble_count = None
        self.dropout = None
        self.hidden_layers = None
        self.neurons = None
        self.verbose = None
        self.epochs = None
        self.batch_size = None
        self.shuffle = None
        self.validation_split = None
        self.callbacks = None
        self.vectorise = None
        self.model_type = None

        if data.is_processed():
            self.data = self._validate(data, ModelData)
        else:
            raise AssertionError('Dataset to use is not properly processed')

        self.ensemble = self._validate(ensemble, bool)
        if self.ensemble:
            raise NotImplementedError('Ensemble models do not work')
            self.ensemble_count = self._validate(ensemble_count, int, range(0, 10000))
        else:
            if ensemble_count is not None:
                logging.warning('\tYou tried to specify the number of ensemble models to instantiate when you set the '
                                'ensemble flag to False. Are you sure you want to proceed?')

    def instantiate(self, hidden_layers: int, neurons_per_layer: Union[Iterable[int], int], dropout_threshold: float,
                    validation_split: float, model_type: str = 'RNN', epochs: int = 10, batch_size: int = 1,
                    shuffle: bool = True, verbose: int = 1, patience: int = 10):
        """
        Method to instantiate the model/models used for the classification task

        Parameters
        ----------
        hidden_layers:       Specifies the number of hidden layers to create in the model
        neurons_per_layer:   Specifies the number of neurons per hidden layer
        dropout_threshold:   Specifies the fraction of neurons to be dropped out per iteration
        model_type:          Specifies the type of model being trained
        epochs:              Number of training iterations
        batch_size:          The fractional split of the overall dataset to use for training per iteration
        shuffle:             Shuffle the dataset during training
        validation_split:    Fraction of dataset to be used for validation
        verbose:             Integer representing the level of logging provided during training
        patience:            The number of training cycles to continue when there is no improvements to the loss
                             or accuracy
        """

        def _instantiate_model(self) -> tf.keras.Sequential:
            """Internal method to instantiate a specified model"""

            # create the word vectorisation layer first
            self.vectorise = TextVectorization(max_tokens=MAX_TOKENS, output_mode='int',
                                               output_sequence_length=MAX_PADDING)
            self.vectorise.adapt(self.data.X)

            if self.model_type != 'CNN-GRU':
                # create model now
                model = Sequential()
                model.add(Input(shape=(1,), dtype=tf.string))
                model.add(self.vectorise)

                if self.model_type == 'Simple':
                    # create the hidden layers
                    for lyr in range(self.hidden_layers):
                        if isinstance(self.neurons, Iterable):
                            model.add(Dense(self.neurons[lyr], activation='relu'))
                        else:
                            model.add(Dense(self.neurons, activation='relu'))
                        model.add(Dropout(self.dropout))
                elif self.model_type == 'RNN':
                    model.add(Embedding(input_dim=len(self.vectorise.get_vocabulary()),
                                        output_dim=self.neurons,
                                        mask_zero=True))
                    model.add(Bidirectional(LSTM(self.neurons)))
                    for lyr in range(self.hidden_layers):
                        if isinstance(self.neurons, Iterable):
                            model.add(Dense(self.neurons[lyr], activation='relu'))
                        else:
                            model.add(Dense(self.neurons, activation='relu'))
                        model.add(Dropout(self.dropout))
                elif self.model_type == 'BiLSTM':
                    model.add(Embedding(input_dim=len(self.vectorise.get_vocabulary()),
                                        output_dim=self.neurons,
                                        mask_zero=True))
                    model.add(Bidirectional(LSTM(self.neurons, return_sequences=True)))
                    model.add(Bidirectional(LSTM(int(self.neurons / 2))))
                    for lyr in range(self.hidden_layers):
                        if isinstance(self.neurons, Iterable):
                            model.add(Dense(self.neurons[lyr], activation='relu'))
                        else:
                            model.add(Dense(self.neurons, activation='relu'))
                        model.add(Dropout(self.dropout))

                model.add(Dense(self.data.y.shape[1], activation='softmax'))

                # compile model and return
                model.compile(optimizer='adam',
                              loss='categorical_crossentropy',
                              metrics=['accuracy'])
                return model
            else:
                raise NotImplementedError('Still in Testing...')
                model1 = Input(shape=(1,))
                vectorise1 = TextVectorization(max_tokens=MAX_TOKENS, output_mode='int',
                                               output_sequence_length=MAX_PADDING)
                vectorise1.adapt(self.data.X)
                vectorise1 = vectorise1(model1)
                embedding1 = Embedding(input_dim=len(self.vectorise.get_vocabulary()),
                                       output_dim=self.neurons,
                                       mask_zero=True)(vectorise1)
                conv1 = Conv1D(filters=100,
                               kernel_size=(4,),
                               activation='relu',
                               kernel_constraint=MaxNorm(max_value=3, axis=[0, 1]))(embedding1)
                pool1 = MaxPool1D(pool_size=2,
                                  strides=2)(conv1)
                flat1 = Flatten()(pool1)
                drop1 = Dropout(self.dropout)(flat1)
                dense1 = Dense(self.neurons, activation='relu')(drop1)
                drop1 = Dropout(self.dropout)(dense1)
                output1 = Dense(self.data.y.shape[1], activation='softmax')(drop1)

                model2 = Input(shape=(1,))
                vectorise2 = TextVectorization(max_tokens=MAX_TOKENS, output_mode='int',
                                               output_sequence_length=MAX_PADDING)
                vectorise2.adapt(self.data.X)
                vectorise2 = vectorise2(model2)
                embedding2 = Embedding(input_dim=len(self.vectorise.get_vocabulary()),
                                       output_dim=self.neurons,
                                       mask_zero=True)(vectorise2)
                bidirectional2 = Bidirectional(GRU(self.neurons))(embedding2)
                drop2 = Dropout(self.dropout)(bidirectional2)
                output2 = Dense(self.data.y.shape[1], activation='softmax')(drop2)

                merged = concatenate([output1, output2])
                output = Dense(self.data.y.shape[1], activation='softmax')(merged)
                model = KerasModel(inputs=[model1, model2], outputs=output)
                model.compile(optimizer='adam',
                              loss='categorical_crossentropy',
                              metrics=['accuracy'])
                return model

        # validate temp vars
        hidden = self._validate(hidden_layers, int, range(1, MAX))
        neurons = self._validate(neurons_per_layer, (Iterable, int))
        dropout = self._validate(dropout_threshold, float)

        # validate dropout value
        if 0 < dropout < 1:
            self.dropout = dropout
        else:
            raise ValueError('dropout must be a float between 0 and 1 (not inclusive)')

        # validate training params
        if isinstance(neurons, Iterable):
            if len(neurons) < hidden:
                # pad to number of hidden layers
                to_pad = hidden - len(neurons)
                final = neurons[-1]
                neurons = neurons + ([final] * to_pad)
                self.hidden_layers = hidden
                self.neurons = neurons
            elif len(neurons) > hidden:
                # truncate
                neurons = neurons[:hidden]
                self.hidden_layers = hidden
                self.neurons = neurons
            else:
                self.hidden_layers = hidden
                self.neurons = neurons
            self.verbose = self._validate(verbose, int, range(0, 3))
            self.epochs = self._validate(epochs, int, range(1, MAX))
            self.batch_size = self._validate(batch_size, int, range(1, MAX))
            self.shuffle = self._validate(shuffle, bool)
            self.model_type = self._validate(model_type, str, ('Simple', 'RNN', 'BiLSTM', 'CNN-GRU'))
            self.validation_split = self._validate(validation_split, float)
            if 0 < validation_split < 1:
                self.validation_split = validation_split
            else:
                raise ValueError('patience must be a float between 0 and 1 (not inclusive)')

        else:
            self.hidden_layers = hidden
            self.neurons = neurons
            self.verbose = self._validate(verbose, int, range(0, 3))
            self.epochs = self._validate(epochs, int, range(1, MAX))
            self.batch_size = self._validate(batch_size, int, range(1, MAX))
            self.shuffle = self._validate(shuffle, bool)
            self.model_type = self._validate(model_type, str, ('Simple', 'RNN', 'BiLSTM', 'CNN-GRU'))
            self.validation_split = self._validate(validation_split, float)
            if 0 < validation_split < 1:
                self.validation_split = validation_split
            else:
                raise ValueError('patience must be a float between 0 and 1 (not inclusive)')

        # set callbacks
        self.callbacks = [
            EarlyStopping(monitor='loss',
                          mode='min',
                          patience=patience,
                          restore_best_weights=True),
            ModelCheckpoint(filepath=f'./models/checkpoints/checkpoints_model_{self.file_counter}',
                            monitor='accuracy',
                            save_weights_only=True,
                            save_best_only=True,
                            save_freq=5)
        ]

        # init models for training
        if self.ensemble:
            self.model = [_instantiate_model(self) for _ in range(self.ensemble_count)]
            display(utils.plot_model(self.model[0], show_shapes=True,
                                     to_file=f'assets/models/model_{self.file_counter}.png',
                                     show_layer_names=True))
        else:
            self.model = _instantiate_model(self)
            display(
                utils.plot_model(self.model, show_shapes=True, to_file=f'assets/models/model_{self.file_counter}.png',
                                 show_layer_names=True))

    def fit(self, persist: bool = True):
        """
        Fits the model(s) stored

        Ensemble fitting is done with fit_ensemble() function

        Parameters
        ----------
        persist:                Flag to indicate whether to persist model to disk or not
        """

        assert self.model is not None
        persist = self._validate(persist, bool)

        if self.model_type != 'CNN-GRU':
            if self.ensemble:
                # train all the submodels first
                self.history = []
                for i in range(len(self.model)):
                    self.history.append(self.model[i].fit(self.data.X_train,
                                                          self.data.y_train,
                                                          epochs=self.epochs,
                                                          batch_size=self.batch_size,
                                                          shuffle=self.shuffle,
                                                          validation_split=self.validation_split,
                                                          verbose=self.verbose,
                                                          callbacks=self.callbacks
                                                          ))
                    if persist:
                        if not os.path.isdir(os.path.join(os.getcwd(), 'models')):
                            os.mkdir(os.path.join(os.getcwd(), 'models'))

                        self.model[i].save(os.path.join(os.getcwd(), 'models', f'model_{self.file_counter}'))
                        self.file_counter += 1
                    logging.info(f'Model {i} successfully trained!')

                # start ensemble training

                logging.info(
                    'All models trained successfully! Beginning with ensemble model instantiation and training...')

            else:
                self.history = self.model.fit(self.data.X_train,
                                              self.data.y_train,
                                              epochs=self.epochs,
                                              batch_size=self.batch_size,
                                              shuffle=self.shuffle,
                                              validation_split=self.validation_split,
                                              verbose=self.verbose,
                                              callbacks=self.callbacks
                                              )
                if persist:
                    if not os.path.isdir(os.path.join(os.getcwd(), 'models')):
                        os.mkdir(os.path.join(os.getcwd(), 'models'))

                    self.model.save(os.path.join(os.getcwd(), 'models', f'model_{self.file_counter}'))
                    self.file_counter += 1
                logging.info(f'Model successfully trained!')
        else:
            self.history = self.model.fit([self.data.X_train, self.data.X_train],
                                          self.data.y_train,
                                          epochs=self.epochs,
                                          batch_size=self.batch_size,
                                          shuffle=self.shuffle,
                                          validation_split=self.validation_split,
                                          verbose=self.verbose,
                                          callbacks=self.callbacks
                                          )
            if persist:
                if not os.path.isdir(os.path.join(os.getcwd(), 'models')):
                    os.mkdir(os.path.join(os.getcwd(), 'models'))

                self.model.save(os.path.join(os.getcwd(), 'models', f'model_{self.file_counter}'))
                self.file_counter += 1
            logging.info(f'Model successfully trained!')

    def fit_ensemble(self, neuron: int, validation_split: float, epochs: int = 10, batch_size: int = 1,
                     shuffle: bool = True, verbose: int = 1, patience: int = 10,
                     persist: bool = True) -> None:
        """
        Fits the ensemble model by first turning all layers in the model untrainable, and then constructing
        the model from scratch

        This function accepts similar inputs as the above .instantiate() function

        Parameters
        ----------
        neuron:              Number of neurons in the sole layer in ensemble layer
        epochs:              Number of training iterations
        batch_size:          The fractional split of the overall dataset to use for training per iteration
        shuffle:             Shuffle the dataset during training
        validation_split:    Fraction of dataset to be used for validation
        verbose:             Integer representing the level of logging provided during training
        patience:            The number of training cycles to continue when there is no improvements to the loss
                             or accuracy
        persist:             Persist model to disk if set to True
        """

        if not self.ensemble or not isinstance(self.model, Iterable):
            raise AssertionError('Model stored is not compatible with Ensemble training')

        # validate all params
        neuron = self._validate(neuron, int, range(1, MAX))
        epochs = self._validate(epochs, int, range(1, MAX))
        batch_size = self._validate(batch_size, int, range(1, MAX))
        shuffle = self._validate(shuffle, bool)
        verbose = self._validate(verbose, int, range(0, 3))
        patience = self._validate(patience, int, range(1, MAX))
        persist = self._validate(persist, bool)
        validation_split = self._validate(validation_split, float)
        if 0 < validation_split < 1:
            self.validation_split = validation_split
        else:
            raise ValueError('patience must be a float between 0 and 1 (not inclusive)')

        # set callbacks
        self.callbacks = [
            EarlyStopping(monitor='loss',
                          mode='min',
                          patience=patience,
                          restore_best_weights=True),
            ModelCheckpoint(filepath=f'./models/checkpoints/checkpoints_model_{self.file_counter}',
                            monitor='accuracy',
                            save_weights_only=True,
                            save_best_only=True,
                            save_freq=5)
        ]

        def _init_ensemble(self) -> tf.keras.Model:
            """Internal helper function to create the ensemble model using the same specs as the trained model"""

            for i in range(len(self.model)):
                for lyr in self.model[i].layers:
                    lyr.trainable = False
                    lyr._name = f'ensemble_{i}_{lyr.name}'
            _in = [mdl.input for mdl in self.model]
            _out = [mdl.output for mdl in self.model]

            # merge all layers
            merged = concatenate(_out)
            hidden_activations = Dense(neuron, activation='relu')(merged)
            output = Dense(self.data.y.shape[1], activation='softmax')(hidden_activations)
            model = KerasModel(inputs=_in, outputs=output)
            display(utils.plot_model(model, show_shapes=True,
                                     to_file=f'assets/models/ensemble_model_{self.file_counter}.png'))
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            return model

        self.ensemble_model = _init_ensemble(self)

        # fit the ensemble model
        X = [self.data.X_train for _ in range(len(self.ensemble_model.input))]
        self.ensemble_model.fit(X,
                                self.data.y,
                                epochs=epochs,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                validation_split=validation_split,
                                verbose=verbose,
                                callbacks=self.callbacks)

        if persist:
            if not os.path.isdir(os.path.join(os.getcwd(), 'models')):
                os.mkdir(os.path.join(os.getcwd(), 'models'))

            self.ensemble_model.save(os.path.join(os.getcwd(), 'models', f'ensemble_model_{self.file_counter}'))
            self.file_counter += 1

    def evaluate(self, batch_size: int) -> None:
        """
        Evaluates the accuracy of the model

        Parameters
        ----------
        batch_size:          The fractional split of the overall dataset to use for evaluation
        """

        batch_size = self._validate(batch_size, int, range(1, MAX))

        if self.ensemble:
            sub_models = [sub.evaluate(self.data.X_test, self.data.y_test, batch_size=batch_size) for sub in
                          self.model]
            print(f'Submodels Accuracy: {sub_models}')

            if self.ensemble_model is not None:
                ensemble = self.ensemble_model.evaluate([self.data.X_test for _ in range(self.ensemble_count)],
                                                        self.data.y_test, batch_size=batch_size)
                print(f'Ensemble Accuracy: {ensemble}')
            else:
                logging.warning('Ensemble Training has not been conducted yet, hence no ensemble model '
                                'accuracy can be shown.')
        else:
            if self.model is None:
                logging.warning('No models have been trained yet')
            else:
                single = self.model.evaluate(self.data.X_test, self.data.y_test, batch_size=batch_size)
                print(f'Single Model Accuracy: {single}')

    def predict(self, to_predict: str, interpret: Optional[Callable] = None) -> np.array:
        """
        Predicts the label based on the input string

        Parameters
        ----------
        to_predict:         String to feed into model for predictions
        interpret:          Optional function to interpret predictions
        """

        to_predict = [self._validate(to_predict, str)]

        if self.ensemble:
            if self.ensemble_model is not None:
                data = tuple(to_predict for _ in range(len(self.ensemble_model.input)))
                print(data)
                predictions = self.ensemble_model.predict(data)
            else:
                logging.warning('Ensemble Training has not been conducted yet.')
        else:
            if self.model is not None:
                predictions = self.model.predict(to_predict)
            else:
                logging.warning('No models have been trained yet.')

        if interpret is not None:
            return interpret(predictions)

        return self.data.encoder.inverse_transform([np.argmax(predictions)])

    @staticmethod
    def _validate(value: Any, value_type: Any, value_range: Optional[Iterable] = None) -> Any or None:
        """
        Validates the input value's type and optionally the valid range of permitted values

        Parameters
        ----------
        value:           Any value to test
        value_type:      Any type to test, either contained within an interable like a tuple or as a single type
        value_range:     Iterable containing Any type

        Raises
        ------
        ValueError:      If value is not found in value_range
        TypeError:       If type of value is not value_type
        """

        if isinstance(value, value_type):
            if value_range is not None:
                value_range = [v for v in value_range]
                if value in value_range:
                    return value
                else:
                    raise ValueError(f'{value} is out of range')
            else:
                return value
        else:
            raise TypeError(f'{value} not of type {value_type}')