import numpy as np
import tensorflow as tf
import keras_tuner as kt
from config import GLOBALS

from data import *
from typing import *
from tensorflow.keras import utils
from tensorflow.keras import Model as KerasModel
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential, load_model
from keras.layers import Dense, TextVectorization, Dropout, concatenate, Input, Embedding, \
    Bidirectional, LSTM, GRU, MaxPool1D, Flatten, Conv1D, Average, Maximum, Add
from keras.constraints import MaxNorm
from keras.callbacks import EarlyStopping, ModelCheckpoint

# read config file first and set constants
MAX = GLOBALS['GLOBAL_MAX']
MAX_TOKENS = GLOBALS['MAX_TOKENS']
MAX_PADDING = GLOBALS['MAX_PADDING']


class Model:
    """This Class contains methods used for training a standalone or ensemble model to use to classify tokens"""

    def __init__(self, m_data: ModelData, ensemble: bool = True, ensemble_count: Optional[int] = None):
        """
        Initialises the MLP Model Class

        Parameters
        ----------
        m_data:             ModelData instance containing all the data to use for training
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
        self.patience = None
        self.tuner = None
        self.to_map = False

        if m_data.is_processed():
            self.data = self._validate(m_data, ModelData)
        else:
            raise AssertionError('Dataset to use is not properly processed')

        self.ensemble = self._validate(ensemble, bool)
        if self.ensemble:
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

        # validate temp vars
        hidden = self._validate(hidden_layers, int, range(1, MAX))
        neurons = self._validate(neurons_per_layer, (Iterable, int))

        # validate training params
        if isinstance(neurons, int):
            self.to_map = False
            self.hidden_layers = hidden
            self.neurons = neurons
        elif isinstance(neurons, Iterable) and not isinstance(neurons, str):
            self.to_map = True
            if len(neurons) < hidden:
                # pad to number of hidden layers
                to_pad = hidden - len(neurons)
                neurons = neurons + ([neurons[-1]] * to_pad)
                self.hidden_layers = hidden
                self.neurons = neurons
            elif len(neurons) > hidden:
                # truncate
                neurons = neurons[:hidden]
                self.hidden_layers = hidden
                self.neurons = neurons
            else:
                # accept as is
                self.hidden_layers = hidden
                self.neurons = neurons
        else:
            raise ValueError('Neuron count input is invalid')

        self.dropout = self._validate(dropout_threshold, float, normalize=True)
        self.verbose = self._validate(verbose, int, range(0, 3))
        self.epochs = self._validate(epochs, int, range(1, MAX))
        self.batch_size = self._validate(batch_size, int, range(1, MAX))
        self.shuffle = self._validate(shuffle, bool)
        self.model_type = self._validate(model_type, str, ('Simple', 'RNN', 'BiLSTM'))
        self.validation_split = self._validate(validation_split, float, normalize=True)

        # set callbacks
        self.callbacks = [
            EarlyStopping(monitor='val_loss',
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
            utils.plot_model(self.model[0], show_shapes=True,
                             to_file=f'assets/models/model_{self.file_counter}.png',
                             show_layer_names=True)
        else:
            self.model = _instantiate_model(self)
            utils.plot_model(self.model, show_shapes=True, to_file=f'assets/models/model_{self.file_counter}.png',
                             show_layer_names=True)

    def optimise(self, min_neuron: Optional[int] = None, max_neuron: Optional[int] = None,
                 step: Optional[int] = None, dropout: Optional[Iterable[float]] = None, validation_split: float = 0.1,
                 learning_rate: Optional[Iterable] = None, model_type: str = 'RNN', epochs: int = 10,
                 batch_size: int = 1,
                 shuffle: bool = True, verbose: int = 1, patience: int = 10, persist: bool = True):
        """
        This function is an alternative to the fit function that is used to optimise the model.

        Note that this function does not allow you to specify the number of neurons per layer. If you
        want to do that, use .fit() instead.

        WARNING: NOT TESTED

        Parameters
        ----------
        Optimization Possible
        <------------------->
        min_neuron:                 Minimum number of neurons in NN
        max_neuron:                 Maximum number of neurons in NN
        step:                       Increments of neuron number
        learning_rate:              Iterable of floats to use as learning rate
        dropout:                    Iterable of floats to use to test level of dropout

        Hard-coded
        <-------->
        validation_split:           Fraction of dataset to be used for validation
        model_type:                 Specifies the type of model being trained
        epochs:                     Number of training iterations
        batch_size:                 The fractional split of the overall dataset to use for training per iteration
        shuffle:                    Shuffle the dataset during training
        verbose:                    Integer representing the level of logging provided during training
        patience:                   The number of training cycles to continue when there is no improvements to the loss
                                    or accuracy

        Others
        <---->
        persist:                    Flag to save model to disk
        """

        # WARNING
        logging.warning('This function has not been properly tested')

        persist = self._validate(persist, bool)
        min_neuron = self._validate(min_neuron, (type(None), int))
        max_neuron = self._validate(max_neuron, (type(None), int))
        step = self._validate(step, (type(None), int))
        learning_rate = self._validate(learning_rate, (type(None), Iterable))
        assert type(learning_rate) != str

        self.verbose = self._validate(verbose, int, range(0, 3))
        self.epochs = self._validate(epochs, int, range(1, MAX))
        self.batch_size = self._validate(batch_size, int, range(1, MAX))
        self.shuffle = self._validate(shuffle, bool)
        self.model_type = self._validate(model_type, str, ('Simple', 'RNN', 'BiLSTM', 'CNN-GRU'))
        self.patience = self._validate(patience, int, range(0, MAX))
        self.validation_split = self._validate(validation_split, float)
        if 0 < validation_split < 1:
            self.validation_split = validation_split
        else:
            raise ValueError('patience must be a float between 0 and 1 (not inclusive)')

        dropout = self._validate(dropout, (type(None), Iterable))
        if type(dropout) != str:
            for possible in dropout:
                if possible < 0 or possible > 1:
                    raise ValueError('dropout cannot be <0 or >1')

        if min_neuron > max_neuron:
            raise ValueError('Minimum number of neurons cannot be less than the maximum')
        elif max_neuron % max_neuron != 0 or max_neuron % step != 0:
            raise ValueError('Your max neurons, min neurons and step should have the same '
                             'multiplicative base')

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

        def _tune(self, hypertrainer):
            """
            Internal method to init a hyperoptimization-compatible model

            Parameters
            ----------
            hypertrainer:                   Hyperoptimizer
            """

            # create the word vectorisation layer first
            self.vectorise = TextVectorization(max_tokens=MAX_TOKENS, output_mode='int',
                                               output_sequence_length=MAX_PADDING)
            self.vectorise.adapt(self.data.X)

            # define hyperoptimization layers
            neuron_optimizer = hypertrainer.Int('neuron', min_value=min_neuron, max_value=max_neuron, step=step)
            learning_rate_optimizer = hypertrainer.Choice('learning_rate', values=learning_rate)
            dropout_optimizer = hypertrainer.Choice('dropout', values=dropout)

            if self.model_type != 'CNN-GRU':
                # create model now
                model = Sequential()
                model.add(Input(shape=(1,), dtype=tf.string))
                model.add(self.vectorise)

                if self.model_type == 'Simple':
                    # create the hidden layers
                    for lyr in range(self.hidden_layers):
                        model.add(Dense(neuron_optimizer, activation='relu'))
                        model.add(Dropout(dropout_optimizer))
                elif self.model_type == 'RNN':
                    model.add(Embedding(input_dim=len(self.vectorise.get_vocabulary()),
                                        output_dim=self.neurons,
                                        mask_zero=True))
                    model.add(Bidirectional(LSTM(neuron_optimizer)))
                    for lyr in range(self.hidden_layers):
                        model.add(Dense(neuron_optimizer, activation='relu'))
                        model.add(Dropout(dropout_optimizer))
                elif self.model_type == 'BiLSTM':
                    model.add(Embedding(input_dim=len(self.vectorise.get_vocabulary()),
                                        output_dim=self.neurons,
                                        mask_zero=True))
                    model.add(Bidirectional(LSTM(neuron_optimizer, return_sequences=True)))
                    model.add(Bidirectional(LSTM(int(neuron_optimizer / 2))))
                    for lyr in range(self.hidden_layers):
                        model.add(Dense(neuron_optimizer, activation='relu'))
                        model.add(Dropout(dropout_optimizer))

                model.add(Dense(self.data.y.shape[1], activation='softmax'))

                # compile model and return
                model.compile(optimizer=Adam(learning_rate=learning_rate_optimizer),
                              loss='categorical_crossentropy',
                              metrics=['accuracy'])
                return model
            else:
                # i/o first
                inputs = Input(shape=(1,))
                output = [mdl(inputs) for mdl in self.model]

                # init model
                ensemble_output = Average()(output)
                model = KerasModel(inputs=inputs, outputs=ensemble_output)
                model.compile(optimizer='adam',
                              loss='categorical_crossentropy',
                              metrics=['accuracy'])
                return model

        self.tuner = kt.Hyperband(_tune,
                                  objective='val_accuracy',
                                  max_epochs=10,
                                  factor=3,
                                  directory=f'./models/tuned/model_{self.file_counter}',
                                  project_name=f'model_{self.file_counter}_tuner')
        self.tuner.search(self.data.X_train, self.data.y_train,
                          epochs=self.epochs, validation_split=self.validation_split,
                          callbacks=[self.callbacks], batch_size=self.batch_size,
                          verbose=self.verbose, shuffle=self.shuffle)
        best_hyperparams = self.tuner.get_best_hyperparameters(num_trials=1)[0]
        print(best_hyperparams)
        logging.info('Obtained best hyperparameters')

        self.model = self.tuner.hypermodel.build(best_hyperparams)
        self.history = self.model.fit(self.data.X_train, self.data.y_train,
                                      epochs=self.epochs, validation_split=self.validation_split,
                                      verbose=self.verbose, callbacks=[self.callbacks],
                                      batch_size=self.batch_size, shuffle=self.shuffle)
        val_acc = self.history.history['val_accuracy']
        best_epoch = val_acc.index(max(val_acc)) + 1
        logging.info('Best epoch obtained')

        # reinit model and train again with best epoch
        self.model = self.tuner.hypermodel.build(best_hyperparams)
        self.model.fit(self.data.X_train, self.data.y_train,
                       epochs=best_epoch, validation_split=self.validation_split,
                       verbose=self.verbose, callbacks=[self.callbacks],
                       batch_size=self.batch_size, shuffle=self.shuffle)
        logging.info('Hypermodel successfully trained')

        if persist:
            if not os.path.isdir(os.path.join(os.getcwd(), 'models')):
                os.mkdir(os.path.join(os.getcwd(), 'models'))

            self.model.save(os.path.join(os.getcwd(), 'models', f'optimized_model_{self.file_counter}'))
            self.file_counter += 1
            logging.info('Model saved')

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

        if self.model_type != 'Ensemble':
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

    def fit_ensemble(self, on: str = 'average', persist: bool = True) -> None:
        """
        Instantiates and fits the ensemble model

        This function requires an ensemble of models to be trained first, and for the method of
        ensemble output handling to be specified first

        Parameters
        ----------
        on:                  Method to handle the outputs of all the stacked models
                             > average: all model output tensors are averaged
                             > maximum: only the maximum of the model output tensors are returned
                             > add: adds up all the output tensors, element-wise
        persist:             Persist model to disk if set to True
        """

        if not self.ensemble or not isinstance(self.model, Iterable):
            raise AssertionError('Model stored is not compatible with Ensemble training')

        # validate all params
        on = self._validate(on, str, ('average', 'maximum', 'add'))
        persist = self._validate(persist, bool)

        if isinstance(self.model, (Iterable, Sized)):
            if len(self.model) > 1:
                in_lyr = Input(shape=(1,))
                mdl_out = [mdl(in_lyr) for mdl in self.model]
                if on == 'average':
                    outputs = Average()(mdl_out)
                elif on == 'maximum':
                    outputs = Maximum()(mdl_out)
                elif on == 'add':
                    outputs = Add()(mdl_out)
                self.ensemble_model = KerasModel(inputs=in_lyr, outputs=outputs)
                self.ensemble_model.compile(optimizer='adam',
                                            loss='categorical_crossentropy',
                                            metrics=['accuracy'])
                utils.plot_model(self.ensemble_model, show_shapes=True)
            else:
                raise AssertionError('Number of trained ensemble models cannot be less than or equal to 1')
        else:
            raise AssertionError('Invalid Model saved in class')

    def load(self, path: Union[str, Iterable[str]] or Sized) -> None:
        """
        Loads up a list of models or a single model from a list of paths or a path

        Note that this de-initiates any stored models stored in the Model instance

        Parameters
        ----------
        path:               A str or an Sized or Iterable of str of paths to stored models
        """

        path = self._validate(path, (str, Iterable, Sized))
        try:
            if isinstance(path, str):
                self.model = load_model(path)
            else:
                self.model = [load_model(p) for p in path]
        except (FileNotFoundError, IOError):
            raise ValueError('Path contains invalid paths to models')

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

    def predict(self, to_predict: Union[str, list], interpret: Optional[Callable] = None) -> np.array:
        """
        Predicts the label based on the input string

        Parameters
        ----------
        to_predict:         String to feed into model for predictions
        interpret:          Optional function to interpret predictions
        """

        to_predict = [self._validate(to_predict, (str, list))]

        if self.ensemble:
            if self.ensemble_model is not None:
                predictions = self.ensemble_model.predict([to_predict for _ in range(self.ensemble_count)])
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
    def _validate(value: Any, value_type: Any, value_range: Optional[Iterable] = None,
                  normalize: bool = False) -> Any or None:
        """
        Validates the input value's type and optionally the valid range of permitted values

        Parameters
        ----------
        value:           Any value to test
        value_type:      Any type to test, either contained within an interable like a tuple or as a single type
        value_range:     Iterable containing Any type
        normalize:       Checks if the datatype is between 0 and 1

        Raises
        ------
        ValueError:      If value is not found in value_range or if value not normalized
        TypeError:       If type of value is not value_type or if value is not comparable
        """

        if isinstance(value, value_type):
            if value_range is not None:
                if value in value_range:
                    if normalize:
                        try:
                            if 0 < value < 1:
                                return value
                            else:
                                raise ValueError('value is not between 0 and 1')
                        except TypeError:
                            raise TypeError('value does not support comparison using < or >')
                    return value
                else:
                    raise ValueError(f'{value} is out of range')
            else:
                return value
        else:
            raise TypeError(f'{value} not of type {value_type}')
