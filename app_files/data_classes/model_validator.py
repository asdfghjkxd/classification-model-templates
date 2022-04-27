import datetime
import tensorflow as tf
import keras_tuner as kt

from .data_validator import *
from .config_validator import GLOBALS
from typing import *
from pydantic import *
from tensorflow.keras import utils
from tensorflow.keras import Model as KerasModel
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential, load_model
from keras.layers import Dense, TextVectorization, Dropout, Input, Embedding, \
    Bidirectional, LSTM, Average, Maximum, Add
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard


class ModelTrainer(BaseModel):
    """
    This class allows for the validation and manipulation of model parameters

    Attributes
    ----------
    > Globals
    MAX:                    Maximum global integer
    MAX_TOKENS:             Maximum number of tokens for the text vectorisation layer
    MAX_PADDING:            Maximum number of tokens to pad inputs up to
    USE_TENSORBOARD:        Use of Tensorboard to log training progress

    > Mandatory
    ensemble_count:         Integer number to determine the number of ensemble models to instantiate,
                            0 and 1 means that only one model is created, any other positive numbers
                            create more than one model at a time
    model_data:             ModelInput validator class (encapsulates the data)


    > Optional (to be defined in later methods)
    batch_size:             Integer number for batch sizing for training
    callbacks:              A Sequence of keras.callbacks
    dropout:                Float representing the fraction of dropout for the Dropout layer
    ensemble_model:         Optional Tensorflow Model class for the ensemble model
    epochs:                 Integer number of epochs to train the model for
    file_counter:           Integer file counter to start from
    hidden_layers:          Integer number of hidden Dense layers
    history:                Historical data from training the model
    model:                  Tensorflow Sequential class
    model_type:             String representing the type of model to instantiate and train
    neurons:                Sequence of integers or an integer representing the number of neurons per dense layer
    patience:               Integer number for the number of epochs to continue training after no improvements to
                            the benchmark training statistics
    shuffle:                Boolean flag to determine whether to randomly shuffle the input data
    tuner:                  Tensorflow model optimizer
    validation_split:       Float determining the ratio of train-validation split for training
    vectorise:              pass
    verbose:                Level of debug/printing for Model fitting
    """

    class Config:
        title = 'ModelConfig'
        arbitrary_types_allowed = True
        validate_all = True
        validate_assignment = True
        allow_mutation = True
        smart_union = True

    MAX: conint(ge=0, le=GLOBALS['GLOBAL_MAX']) = GLOBALS['GLOBAL_MAX']
    MAX_PADDING: conint(ge=1, le=GLOBALS['MAX_PADDING']) = GLOBALS['MAX_PADDING']
    MAX_TOKENS: conint(ge=1, le=GLOBALS['MAX_TOKENS']) = GLOBALS['MAX_TOKENS']
    USE_TENSORBOARD: bool = GLOBALS['USE_TENSORBOARD']
    batch_size: Optional[conint(le=MAX, ge=1)] = None
    callbacks: Optional[Sequence[Union[EarlyStopping, ModelCheckpoint]]] = None
    dropout: Optional[confloat(gt=0., lt=1.)] = None
    ensemble_count: conint(strict=True, ge=0, le=MAX)
    ensemble_model: Optional[tf.keras.Sequential] = None
    epochs: Optional[conint(le=MAX, ge=1)] = None
    file_counter: Optional[int] = 0
    hidden_layers: Optional[int] = None
    history: Optional[Union[Sequence[tf.keras.callbacks.History], tf.keras.callbacks.History]] = None
    learning_rate: Optional[Sequence[confloat(gt=0., lt=1.)]]
    max_neuron: Optional[conint(le=MAX, gt=0)]
    min_neuron: Optional[conint(le=MAX, gt=0)]
    model: Optional[Union[tf.keras.Sequential, Sequence[tf.keras.Sequential]]] = None
    model_data: ModelInput
    model_type: Optional[str] = None
    neurons: Optional[Union[Sequence[int], int]] = None
    on: Optional[str] = None
    patience: Optional[int] = None
    persist: Optional[bool] = True
    shuffle: Optional[bool] = None
    step: Optional[conint(le=MAX, gt=0)]
    to_predict: Optional[Sequence[str]] = None
    tuner: Optional[kt.Hyperband] = None
    validation_split: Optional[confloat(gt=0., lt=1.)] = None
    vectorise: Optional[TextVectorization] = None
    verbose: Optional[conint(le=3, ge=0)] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = None
        self.callbacks = None
        self.dropout = None
        self.ensemble_model = None
        self.epochs = None
        self.file_counter = 0
        self.hidden_layers = None
        self.history = None
        self.learning_rate = None
        self.max_neuron = None
        self.min_neuron = None
        self.model = None
        self.model_type = None
        self.neurons = None
        self.on = None
        self.patience = None
        self.persist = None
        self.shuffle = None
        self.step = None
        self.to_predict = None
        self.tuner = None
        self.validation_split = None
        self.vectorise = None
        self.verbose = None

    @root_validator(allow_reuse=True)
    def assert_ensemble(cls, values):
        ensemble, count = values.get('ensemble'), values.get('ensemble_count')

        if ensemble:
            if count is None:
                raise AssertionError('Ensemble Count cannot be None if ensemble mode is enabled')
            elif count <= 0:
                raise ValueError('Ensemble Count cannot be zero or negative')

        return values

    @validator('model_type', allow_reuse=True)
    def assert_model_type(cls, v):
        if v in ['Simple', 'RNN', 'BiLSTM'] or v is None:
            return v
        else:
            raise AssertionError('model_type parameter invalid')

    @root_validator(allow_reuse=True)
    def assert_neurons(cls, values):
        _min, _max = values.get('min_neuron'), values.get('max_neuron')
        if not any(map(lambda x: x is None, (_min, _max))):
            if _min >= _max:
                raise ValueError('min_neuron cannot be greater than or equal to max_neuron')

        return values

    @validator('on', allow_reuse=True)
    def assert_on_ensemble(cls, v):
        if v in ['average', 'maximum', 'add'] or v is None:
            return v
        else:
            raise ValueError('on must of of values ["average", "maximum", "add"]')

    @validator('data', always=True, check_fields=False, allow_reuse=True)
    def assert_processed(cls, v):
        if v.is_processed():
            return v
        else:
            raise AssertionError('Input Data is not properly processed')

    def instantiate(self, hidden_layers: int, neurons_per_layer: Union[Sequence[int], int], dropout_threshold: float,
                    validation_split: float, model_type: str = 'RNN', epochs: int = 10, batch_size: int = 1,
                    shuffle: bool = True, verbose: int = 1, patience: int = 10):
        """
        Method to instantiate the model/models

        Parameters
        ----------
        batch_size:          The fractional split of the overall dataset to use for training per iteration
        dropout_threshold:   Specifies the fraction of neurons to be dropped out per iteration
        epochs:              Number of training iterations
        hidden_layers:       Specifies the number of hidden layers to create in the model
        model_type:          Specifies the type of model being trained
        neurons_per_layer:   Specifies the number of neurons per hidden layer
        patience:            The number of training cycles to continue when there is no improvements to the loss
                             or accuracy
        shuffle:             Shuffle the dataset during training
        validation_split:    Fraction of dataset to be used for validation
        verbose:             Integer representing the level of logging provided during training

        """

        def _instantiate_model() -> tf.keras.Sequential:
            """Internal method to instantiate a specified model"""

            # create the word vectorisation layer first
            self.vectorise = TextVectorization(max_tokens=self.MAX_TOKENS, output_mode='int',
                                               output_sequence_length=self.MAX_PADDING)
            self.vectorise.adapt(self.model_data.X)

            # create model now
            model = Sequential()
            model.add(Input(shape=(1,), dtype=tf.string))
            model.add(self.vectorise)

            if self.model_type == 'Simple':
                # create the hidden layers
                for lyr in range(self.hidden_layers):
                    if isinstance(self.neurons, Sequence):
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
                    if isinstance(self.neurons, Sequence):
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
                    if isinstance(self.neurons, Sequence):
                        model.add(Dense(self.neurons[lyr], activation='relu'))
                    else:
                        model.add(Dense(self.neurons, activation='relu'))
                    model.add(Dropout(self.dropout))

            model.add(Dense(self.model_data.y.shape[1], activation='softmax'))

            # compile model and return
            model.compile(optimizer='adam',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
            return model

        if isinstance(neurons_per_layer, int):
            self.hidden_layers = hidden_layers
            self.neurons = neurons_per_layer
        elif isinstance(neurons_per_layer, Sequence) and not isinstance(neurons_per_layer, str):
            if len(neurons_per_layer) < hidden_layers:
                # pad to number of hidden layers
                to_pad = hidden_layers - len(neurons_per_layer)
                neurons_per_layer = [n for n in neurons_per_layer] + ([neurons_per_layer[-1]] * to_pad)
                self.hidden_layers = hidden_layers
                self.neurons = neurons_per_layer
            elif len(neurons_per_layer) > hidden_layers:
                # truncate
                neurons_per_layer = neurons_per_layer[:hidden_layers]
                self.hidden_layers = hidden_layers
                self.neurons = neurons_per_layer
            else:
                # accept as is
                self.hidden_layers = hidden_layers
                self.neurons = neurons_per_layer
        else:
            raise ValueError('Neuron count input is invalid')

        self.batch_size = batch_size
        self.dropout = dropout_threshold
        self.epochs = epochs
        self.model_type = model_type
        self.shuffle = shuffle
        self.validation_split = validation_split
        self.verbose = verbose

        if self.USE_TENSORBOARD:
            self.callbacks = [
                EarlyStopping(monitor='val_loss',
                              mode='min',
                              patience=patience,
                              restore_best_weights=True),
                ModelCheckpoint(filepath=f'../../models/checkpoints/checkpoints_model_{self.file_counter}',
                                monitor='accuracy',
                                save_weights_only=True,
                                save_best_only=True,
                                save_freq=5),
                TensorBoard(log_dir=f'../../logs/fit/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}',
                            histogram_freq=1)
            ]
        else:
            self.callbacks = [
                EarlyStopping(monitor='val_loss',
                              mode='min',
                              patience=patience,
                              restore_best_weights=True),
                ModelCheckpoint(filepath=f'../../models/checkpoints/checkpoints_model_{self.file_counter}',
                                monitor='accuracy',
                                save_weights_only=True,
                                save_best_only=True,
                                save_freq=5)
            ]

        # init models for training
        if self.ensemble_count and self.ensemble_count > 1:
            self.model = [_instantiate_model() for _ in range(self.ensemble_count)]
            utils.plot_model(self.model[0], show_shapes=True,
                             to_file=f'assets/models/model_{self.file_counter}.png',
                             show_layer_names=True)
        else:
            self.model = _instantiate_model()
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
        > Optimization Possible
        min_neuron:                 Minimum number of neurons in NN
        max_neuron:                 Maximum number of neurons in NN
        step:                       Increments of neuron number
        learning_rate:              Iterable of floats to use as learning rate
        dropout:                    Iterable of floats to use to test level of dropout

        > Hard-coded
        validation_split:           Fraction of dataset to be used for validation
        model_type:                 Specifies the type of model being trained
        epochs:                     Number of training iterations
        batch_size:                 The fractional split of the overall dataset to use for training per iteration
        shuffle:                    Shuffle the dataset during training
        verbose:                    Integer representing the level of logging provided during training
        patience:                   The number of training cycles to continue when there is no improvements to the loss
                                    or accuracy

        > Flags
        persist:                    Flag to save model to disk
        """

        # WARNING
        logging.warning('This function has not been properly tested')

        self.min_neuron = min_neuron
        self.max_neuron: max_neuron
        self.step = step
        self.dropout = dropout
        self.validation_split = validation_split
        self.learning_rate = learning_rate
        self.model_type = model_type
        self.epochs = epochs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.verbose = verbose
        self.patience = patience
        self.persist = persist
        if self.USE_TENSORBOARD:
            self.callbacks = [
                EarlyStopping(monitor='val_loss',
                              mode='min',
                              patience=patience,
                              restore_best_weights=True),
                ModelCheckpoint(filepath=f'../../models/checkpoints/checkpoints_model_{self.file_counter}',
                                monitor='accuracy',
                                save_weights_only=True,
                                save_best_only=True,
                                save_freq=5),
                TensorBoard(log_dir=f'../../logs/fit/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}',
                            histogram_freq=1)
            ]
        else:
            self.callbacks = [
                EarlyStopping(monitor='val_loss',
                              mode='min',
                              patience=patience,
                              restore_best_weights=True),
                ModelCheckpoint(filepath=f'../../models/checkpoints/checkpoints_model_{self.file_counter}',
                                monitor='accuracy',
                                save_weights_only=True,
                                save_best_only=True,
                                save_freq=5)
            ]

        def _tune(hypertrainer):
            """
            Internal method to init a hyperoptimization-compatible model

            Parameters
            ----------
            hypertrainer:                   Hyperoptimizer
            """

            # create the word vectorisation layer first
            self.vectorise = TextVectorization(max_tokens=self.MAX_TOKENS, output_mode='int',
                                               output_sequence_length=self.MAX_PADDING)
            self.vectorise.adapt(self.model_data.X)

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

                model.add(Dense(self.model_data.y.shape[1], activation='softmax'))

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
        self.tuner.search(self.model_data.X_train, self.model_data.y_train,
                          epochs=self.epochs, validation_split=self.validation_split,
                          callbacks=[self.callbacks], batch_size=self.batch_size,
                          verbose=self.verbose, shuffle=self.shuffle)
        best_hyperparams = self.tuner.get_best_hyperparameters(num_trials=1)[0]
        print(best_hyperparams)
        logging.info('Obtained best hyperparameters')

        self.model = self.tuner.hypermodel.build(best_hyperparams)
        self.history = self.model.fit(self.model_data.X_train, self.model_data.y_train,
                                      epochs=self.epochs, validation_split=self.validation_split,
                                      verbose=self.verbose, callbacks=[self.callbacks],
                                      batch_size=self.batch_size, shuffle=self.shuffle)
        val_acc = self.history.history['val_accuracy']
        best_epoch = val_acc.index(max(val_acc)) + 1
        logging.info('Best epoch obtained')

        # reinit model and train again with best epoch
        self.model = self.tuner.hypermodel.build(best_hyperparams)
        self.model.fit(self.model_data.X_train, self.model_data.y_train,
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

        self.persist = persist
        assert self.model is not None

        if self.model_type != 'Ensemble':
            if self.ensemble_count > 1:
                # train all the submodels first
                history = []
                for i in range(len(self.model)):
                    history.append(self.model[i].fit(self.model_data.X_train,
                                                     self.model_data.y_train,
                                                     epochs=self.epochs,
                                                     batch_size=self.batch_size,
                                                     shuffle=self.shuffle,
                                                     validation_split=self.validation_split,
                                                     verbose=self.verbose,
                                                     callbacks=self.callbacks
                                                     ))
                    if self.persist:
                        if not os.path.isdir(os.path.join(os.getcwd(), 'models')):
                            os.mkdir(os.path.join(os.getcwd(), 'models'))

                        self.model[i].save(os.path.join(os.getcwd(), 'models', f'model_{self.file_counter}'))
                        self.file_counter += 1
                    logging.info(f'Model {i} successfully trained!')
                self.history = history
                logging.info('All models trained successfully! Beginning with ensemble model instantiation '
                             'and training...')
            else:
                self.history = self.model.fit(self.model_data.X_train,
                                              self.model_data.y_train,
                                              epochs=self.epochs,
                                              batch_size=self.batch_size,
                                              shuffle=self.shuffle,
                                              validation_split=self.validation_split,
                                              verbose=self.verbose,
                                              callbacks=self.callbacks
                                              )
                if self.persist:
                    if not os.path.isdir(os.path.join(os.getcwd(), 'models')):
                        os.mkdir(os.path.join(os.getcwd(), 'models'))

                    self.model.save(os.path.join(os.getcwd(), 'models', f'model_{self.file_counter}'))
                    self.file_counter += 1
                logging.info(f'Model successfully trained!')
        else:
            self.history = self.model.fit([self.model_data.X_train, self.model_data.X_train],
                                          self.model_data.y_train,
                                          epochs=self.epochs,
                                          batch_size=self.batch_size,
                                          shuffle=self.shuffle,
                                          validation_split=self.validation_split,
                                          verbose=self.verbose,
                                          callbacks=self.callbacks
                                          )
            if self.persist:
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

        if not self.ensemble_count or not isinstance(self.model, Iterable):
            raise AssertionError('Model stored is not compatible with Ensemble training')

        self.on = on
        self.persist = persist

        if isinstance(self.model, Sequence):
            if len(self.model) > 1:
                in_lyr = Input(shape=(1,))
                mdl_out = [mdl(in_lyr) for mdl in self.model]
                if self.on == 'average':
                    outputs = Average()(mdl_out)
                    self.ensemble_model = KerasModel(inputs=in_lyr, outputs=outputs)
                elif self.on == 'maximum':
                    outputs = Maximum()(mdl_out)
                    self.ensemble_model = KerasModel(inputs=in_lyr, outputs=outputs)
                elif self.on == 'add':
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

    def load(self, path: Union[str, Sequence[str], Sequence[os.PathLike]]) -> None:
        """
        Loads up a list of models or a single model from a list of paths or a path

        Note that this de-initiates any stored models stored in the Model instance

        Parameters
        ----------
        path:               A str or an Sized or Iterable of str of paths to stored models
        """

        try:
            if isinstance(path, str):
                self.model = load_model(path)
            else:
                self.model = [load_model(p) for p in path]
        except (FileNotFoundError, IOError):
            raise ValueError('Path contains invalid paths to models')

    def evaluate(self, batch_size: int) -> list:
        """
        Evaluates the accuracy of the model

        Parameters
        ----------
        batch_size:          The fractional split of the overall dataset to use for evaluation
        """

        self.batch_size = batch_size
        if self.ensemble_count and self.ensemble_count > 1:
            assert self.ensemble_model is not None, 'Ensemble Training has not been conducted yet, hence no ensemble ' \
                                                    'model accuracy can be shown.'

            sub_models = [sub.evaluate(self.model_data.X_test, self.model_data.y_test,
                                       batch_size=batch_size) for sub in self.model]
            ensemble = self.ensemble_model.evaluate([self.model_data.X_test for _ in range(self.ensemble_count)],
                                                    self.model_data.y_test, batch_size=batch_size)
            print(f'Sub-model Accuracies: {sub_models}\nEnsemble Accuracy: {ensemble}')
            return [sub_models, ensemble]
        else:
            assert self.model is not None, 'No models have been trained yet'

            single = self.model.evaluate(self.model_data.X_test, self.model_data.y_test, batch_size=batch_size)
            print(f'Single Model Accuracy: {single}')
            return [single]

    def predict(self, to_predict: Union[str, list], interpret: Optional[Callable] = None) -> np.array:
        """
        Predicts the label based on the input string

        Parameters
        ----------
        to_predict:         String to feed into model for predictions
        interpret:          Optional function to interpret predictions
        """

        self.to_predict = [to_predict]

        if self.ensemble_count > 1:
            assert self.ensemble_model is not None, 'Ensemble Training has not been conducted yet.'
            predictions = self.ensemble_model.predict([to_predict for _ in range(self.ensemble_count)])

            if interpret is not None:
                return interpret(predictions)

            return self.model_data.encoder.inverse_transform([np.argmax(predictions)])
        else:
            assert self.model is not None, 'No models have been trained yet.'
            predictions = self.model.predict(to_predict)

            if interpret is not None:
                return interpret(predictions)

            return self.model_data.encoder.inverse_transform([np.argmax(predictions)])


if __name__ == '__main__':
    mdl = ModelInput(path=r'C:\Users\User\Documents\Github\Planner\assets\sample_files\test_classification.csv',
                     format='csv', on_error='raise')
    trainer = ModelTrainer(model_data=mdl, ensemble_count=0)
