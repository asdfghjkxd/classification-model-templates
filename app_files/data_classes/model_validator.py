import logging
import numpy as np
import tensorflow as tf
import keras_tuner as kt
import abc


from config_utils.utils import *
from config_utils.config import GLOBALS
from data_validator import *
from typing import *
from pydantic import *
from tensorflow.keras import utils
from tensorflow.keras import Model as KerasModel
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential, load_model
from keras.layers import Dense, TextVectorization, Dropout, Input, Embedding, \
    Bidirectional, LSTM, Average, Maximum, Add
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, \
    TrainingArguments, Trainer, create_optimizer


# +-------------------------------------------------------------------------------------------------------------------+
# |                                     ABSTRACT BASE CLASS FOR ALL MODEL CLASSES                                     |
# +-------------------------------------------------------------------------------------------------------------------+
class AbstractBaseModel(BaseModel, abc.ABC):
    """
    This defines the basic attribute and functional contract that all model classes must implement to function
    
    All child models must possess the following attributes:
    
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


    > Faux-private (to be defined in later methods)
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


    All child models must also implement the following methods as enforced in the contract

    Methods
    -------
    __init__:               Instantiates the class and provide defaults for the class attributes
    instantiate:            Instantiates the model
    optimise:               Optional Model optimisation function
    fit:                    Mandatory Model fitting function
    fit_ensemble:           Optional Ensemble Model fitting function
    load:                   Loads up a trained model from disk
    evaluate:               Evaluates the performance of the loaded model
    predict:                Uses the trained model to conduct predictions on novel data
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
    callbacks: Optional[Sequence[Union[EarlyStopping, ModelCheckpoint, TensorBoard]]] = None
    dropout: Optional[Union[confloat(gt=0., lt=1.), Sequence[confloat(gt=0., lt=1.)]]] = None
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
    model_data: Union[ModelData, BERTModelData]
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

    @classmethod
    @root_validator(allow_reuse=True)
    def assert_ensemble(cls, values):
        ensemble, count = values.get('ensemble'), values.get('ensemble_count')

        if ensemble:
            if count is None:
                raise AssertionError('Ensemble Count cannot be None if ensemble mode is enabled')
            elif count <= 0:
                raise ValueError('Ensemble Count cannot be zero or negative')

        return values

    @classmethod
    @validator('model_type', allow_reuse=True)
    def assert_model_type(cls, v):
        if v in ['Simple', 'RNN', 'BiLSTM', 'Base Uncased BERT', 'Large Uncased BERT',
                 'Base Cased Bert', 'Large Cased Bert'] or v is None:
            return v
        else:
            raise AssertionError('model_type parameter invalid')

    @classmethod
    @root_validator(allow_reuse=True)
    def assert_neurons(cls, values):
        _min, _max = values.get('min_neuron'), values.get('max_neuron')
        if not any(map(lambda x: x is None, (_min, _max))):
            if _min >= _max:
                raise ValueError('min_neuron cannot be greater than or equal to max_neuron')

        return values

    @classmethod
    @validator('on', allow_reuse=True)
    def assert_on_ensemble(cls, v):
        if v in ['average', 'maximum', 'add'] or v is None:
            return v
        else:
            raise ValueError('on must of of values ["average", "maximum", "add"]')

    @classmethod
    @validator('data', always=True, check_fields=False, allow_reuse=True)
    def assert_processed(cls, v):
        if v.is_processed():
            return v
        else:
            raise AssertionError('Input Data is not properly processed')

    @abc.abstractmethod
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

    @abc.abstractmethod
    def instantiate(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def optimise(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
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
            logging.info(f'All models successfully trained!')
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

    @abc.abstractmethod
    def fit_ensemble(self, on: str = 'average', persist: bool = True):
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

        if not self.ensemble_count or not isinstance(self.model, Sequence):
            raise AssertionError('Model stored is not compatible with Ensemble training')
        elif len(self.model) <= 1 or self.ensemble_model <= 1:
            raise AssertionError('Number of trained ensemble models cannot be less than or equal to 1')

        self.on = on
        self.persist = persist

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
        self.ensemble_model.fit(self.model_data.X_train,
                                self.model_data.y_train,
                                epochs=self.epochs,
                                batch_size=self.batch_size,
                                shuffle=self.shuffle,
                                validation_split=self.validation_split,
                                verbose=self.verbose,
                                callbacks=self.callbacks)

        if self.persist:
            if not os.path.isdir(os.path.join(os.getcwd(), 'models')):
                os.mkdir(os.path.join(os.getcwd(), 'models'))

            self.ensemble_model.save(os.path.join(os.getcwd(), 'models', f'ensemble_model_{self.file_counter}'))
            self.file_counter += 1
        logging.info(f'Ensemble model successfully trained!')

    @abc.abstractmethod
    def load(self, path: Union[str, Sequence[str], Sequence[os.PathLike]]):
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

    @abc.abstractmethod
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

    @abc.abstractmethod
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


class AbstractBERTModel(AbstractBaseModel):
    bert_vectorizer: Optional[Any] = None
    bert_data_collator: Optional[DataCollatorWithPadding] = None

    @abc.abstractmethod
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ensemble_model = 1
        self.bert_vectorizer = None
        self.bert_data_collator = None

    @abc.abstractmethod
    def instantiate(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def optimise(self, *args, **kwargs):
        logging.warning('BERT Models do not support Ensemble Models')
        return None

    @abc.abstractmethod
    def fit(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def fit_ensemble(self, *args, **kwargs):
        logging.warning('BERT Models do not support Ensemble Models')
        return None

    @abc.abstractmethod
    def load(self, *args, **kwargs):
        logging.warning('BERT Models do not support Ensemble Models')
        return None

    @abc.abstractmethod
    def evaluate(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def predict(self, *args, **kwargs):
        pass

    def preprocessor(self, example):
        """Preprocessing function"""

        return self.bert_tokenizer(example, trunctation=True)


# +-------------------------------------------------------------------------------------------------------------------+
# |                                        CONCRETE CLASS FOR ALL MODEL CLASSES                                       |
# +-------------------------------------------------------------------------------------------------------------------+
class SimpleModel(AbstractBaseModel):
    """Simple Neural Networks consisting of only Dense and Dropout layers"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def instantiate(self,
                    batch_size: int = 1,
                    dropout_threshold: float = 0.8,
                    epochs: int = 10,
                    hidden_layers: int = 1,
                    neurons_per_layer: Union[Sequence[int], int] = 64,
                    patience: int = 10,
                    shuffle: bool = True,
                    validation_split: float = 0.8,
                    verbose: int = 1):
        """
        Instantiates a Simple Neural Network for training

        Parameters
        ----------
        batch_size:          The fractional split of the overall dataset to use for training per iteration
        dropout_threshold:   Specifies the fraction of neurons to be dropped out per iteration
        epochs:              Number of training iterations
        hidden_layers:       Specifies the number of hidden layers to create in the model
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

            model = Sequential()
            model.add(Input(shape=(1,), dtype=tf.string))
            model.add(self.vectorise)

            # create the hidden layers
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
        self.shuffle = shuffle
        self.validation_split = validation_split
        self.verbose = verbose
        self.patience = patience

        # set callbacks for the entire training function
        set_callbacks(self)

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

    def optimise(self,
                 min_neuron: int = 8,
                 max_neuron: int = 64,
                 step: int = 8,
                 dropout: Sequence[float] = (0.3, 0.5, 0.7),
                 validation_split: float = 0.1,
                 learning_rate: Sequence = (1e-3, 1e-4, 1e-5),
                 epochs: int = 10,
                 batch_size: int = 1,
                 shuffle: bool = True,
                 verbose: int = 1,
                 patience: int = 10,
                 persist: bool = True):
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
        self.epochs = epochs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.verbose = verbose
        self.patience = patience
        self.persist = persist

        # set callbacks for the entire optimization function
        set_callbacks(self)

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

            # create model now
            model = Sequential()
            model.add(Input(shape=(1,), dtype=tf.string))
            model.add(self.vectorise)
            for lyr in range(self.hidden_layers):
                model.add(Dense(neuron_optimizer, activation='relu'))
                model.add(Dropout(dropout_optimizer))
            model.add(Dense(self.model_data.y.shape[1], activation='softmax'))

            # compile model and return
            model.compile(optimizer=Adam(learning_rate=learning_rate_optimizer),
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
        Fits the model(s) stored, using the abstract method defined in abstract base class

        Ensemble fitting is done with fit_ensemble() function

        Parameters
        ----------
        persist:                Flag to indicate whether to persist model to disk or not
        """

        super().fit(persist=persist)

    def fit_ensemble(self, on: str = 'average', persist: bool = True):
        """
        Instantiates and fits the ensemble model, using the abstract method defined in
        abstract base class

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

        super().fit_ensemble(on=on, persist=persist)

    def load(self, path: Union[str, Sequence[str], Sequence[os.PathLike]]):
        """
        Loads up a list of models or a single model from a list of paths or a path,
        implemented in ABC and is called directly from it

        Note that this de-initiates any stored models stored in the Model instance

        Parameters
        ----------
        path:               A str or an Sized or Iterable of str of paths to stored models
        """

        super().load(path=path)

    def evaluate(self, batch_size: int) -> list:
        """
        Evaluates the accuracy of the model, implemented in ABC and is called directly from it

        Parameters
        ----------
        batch_size:          The fractional split of the overall dataset to use for evaluation
        """

        return super().evaluate(batch_size=batch_size)

    def predict(self, to_predict: Union[str, list], interpret: Optional[Callable] = None):
        """
        Predicts the label based on the input string

        Parameters
        ----------
        to_predict:         String to feed into model for predictions
        interpret:          Optional function to interpret predictions
        """

        return super().predict(to_predict=to_predict, interpret=interpret)


class RNNModel(AbstractBaseModel):
    """An RNN model using LSTM, Dense and Dropout layers"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def instantiate(self,
                    batch_size: int = 1,
                    dropout_threshold: float = 0.8,
                    epochs: int = 10,
                    hidden_layers: int = 1,
                    neurons_per_layer: Union[Sequence[int], int] = 64,
                    patience: int = 10,
                    shuffle: bool = True,
                    validation_split: float = 0.8,
                    verbose: int = 1):
        """
        Instantiates a Simple Neural Network for training

        Parameters
        ----------
        batch_size:          The fractional split of the overall dataset to use for training per iteration
        dropout_threshold:   Specifies the fraction of neurons to be dropped out per iteration
        epochs:              Number of training iterations
        hidden_layers:       Specifies the number of hidden layers to create in the model
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

            model = Sequential()
            model.add(Input(shape=(1,), dtype=tf.string))
            model.add(self.vectorise)

            # create the hidden layers
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
        self.shuffle = shuffle
        self.validation_split = validation_split
        self.verbose = verbose
        self.patience = patience

        # set callbacks for the entire training function
        set_callbacks(self)

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

    def optimise(self,
                 min_neuron: int = 8,
                 max_neuron: int = 64,
                 step: int = 8,
                 dropout: Sequence[float] = (0.3, 0.5, 0.7),
                 validation_split: float = 0.1,
                 learning_rate: Sequence = (1e-3, 1e-4, 1e-5),
                 epochs: int = 10,
                 batch_size: int = 1,
                 shuffle: bool = True,
                 verbose: int = 1,
                 patience: int = 10,
                 persist: bool = True):
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
        self.epochs = epochs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.verbose = verbose
        self.patience = patience
        self.persist = persist

        # set callbacks for the entire optimization function
        set_callbacks(self)

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

            # create model now
            model = Sequential()
            model.add(Input(shape=(1,), dtype=tf.string))
            model.add(self.vectorise)
            model.add(Embedding(input_dim=len(self.vectorise.get_vocabulary()),
                                output_dim=self.neurons,
                                mask_zero=True))
            model.add(Bidirectional(LSTM(neuron_optimizer)))
            for lyr in range(self.hidden_layers):
                model.add(Dense(neuron_optimizer, activation='relu'))
                model.add(Dropout(dropout_optimizer))
            model.add(Dense(self.model_data.y.shape[1], activation='softmax'))

            # compile model and return
            model.compile(optimizer=Adam(learning_rate=learning_rate_optimizer),
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
        Fits the model(s) stored, using the abstract method defined in abstract base class

        Ensemble fitting is done with fit_ensemble() function

        Parameters
        ----------
        persist:                Flag to indicate whether to persist model to disk or not
        """

        super().fit(persist=persist)

    def fit_ensemble(self, on: str = 'average', persist: bool = True):
        """
        Instantiates and fits the ensemble model, using the abstract method defined in
        abstract base class

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

        super().fit_ensemble(on=on, persist=persist)

    def load(self, path: Union[str, Sequence[str], Sequence[os.PathLike]]):
        """
        Loads up a list of models or a single model from a list of paths or a path,
        implemented in ABC and is called directly from it

        Note that this de-initiates any stored models stored in the Model instance

        Parameters
        ----------
        path:               A str or an Sized or Iterable of str of paths to stored models
        """

        super().load(path=path)

    def evaluate(self, batch_size: int) -> list:
        """
        Evaluates the accuracy of the model, implemented in ABC and is called directly from it

        Parameters
        ----------
        batch_size:          The fractional split of the overall dataset to use for evaluation
        """

        return super().evaluate(batch_size=batch_size)

    def predict(self, to_predict: Union[str, list], interpret: Optional[Callable] = None):
        """
        Predicts the label based on the input string

        Parameters
        ----------
        to_predict:         String to feed into model for predictions
        interpret:          Optional function to interpret predictions
        """

        return super().predict(to_predict=to_predict, interpret=interpret)


class BidirectionalRNNModel(AbstractBaseModel):
    """A Bidirectional RNN model using multiple LSTM, Dense and Dropout layers"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def instantiate(self,
                    batch_size: int = 1,
                    dropout_threshold: float = 0.8,
                    epochs: int = 10,
                    hidden_layers: int = 1,
                    neurons_per_layer: Union[Sequence[int], int] = 64,
                    patience: int = 10,
                    shuffle: bool = True,
                    validation_split: float = 0.8,
                    verbose: int = 1):
        """
        Instantiates a Simple Neural Network for training

        Parameters
        ----------
        batch_size:          The fractional split of the overall dataset to use for training per iteration
        dropout_threshold:   Specifies the fraction of neurons to be dropped out per iteration
        epochs:              Number of training iterations
        hidden_layers:       Specifies the number of hidden layers to create in the model
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

            model = Sequential()
            model.add(Input(shape=(1,), dtype=tf.string))
            model.add(self.vectorise)

            # create the hidden layers
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
        self.shuffle = shuffle
        self.validation_split = validation_split
        self.verbose = verbose
        self.patience = patience

        # set callbacks for the entire training function
        set_callbacks(self)

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

    def optimise(self,
                 min_neuron: int = 8,
                 max_neuron: int = 64,
                 step: int = 8,
                 dropout: Sequence[float] = (0.3, 0.5, 0.7),
                 validation_split: float = 0.1,
                 learning_rate: Sequence = (1e-3, 1e-4, 1e-5),
                 epochs: int = 10,
                 batch_size: int = 1,
                 shuffle: bool = True,
                 verbose: int = 1,
                 patience: int = 10,
                 persist: bool = True):
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
        self.epochs = epochs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.verbose = verbose
        self.patience = patience
        self.persist = persist

        # set callbacks for the entire optimization function
        set_callbacks(self)

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

            # define hyperoptimization layers1
            neuron_optimizer = hypertrainer.Int('neuron', min_value=min_neuron, max_value=max_neuron, step=step)
            learning_rate_optimizer = hypertrainer.Choice('learning_rate', values=learning_rate)
            dropout_optimizer = hypertrainer.Choice('dropout', values=dropout)

            # create model now
            model = Sequential()
            model.add(Input(shape=(1,), dtype=tf.string))
            model.add(self.vectorise)
            model.add(Embedding(input_dim=len(self.vectorise.get_vocabulary()),
                                output_dim=self.neurons,
                                mask_zero=True))
            model.add(Bidirectional(LSTM(neuron_optimizer)))
            for lyr in range(self.hidden_layers):
                model.add(Dense(neuron_optimizer, activation='relu'))
                model.add(Dropout(dropout_optimizer))
            model.add(Dense(self.model_data.y.shape[1], activation='softmax'))

            # compile model and return
            model.compile(optimizer=Adam(learning_rate=learning_rate_optimizer),
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
        Fits the model(s) stored, using the abstract method defined in abstract base class

        Ensemble fitting is done with fit_ensemble() function

        Parameters
        ----------
        persist:                Flag to indicate whether to persist model to disk or not
        """

        super().fit(persist=persist)

    def fit_ensemble(self, on: str = 'average', persist: bool = True):
        """
        Instantiates and fits the ensemble model, using the abstract method defined in
        abstract base class

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

        super().fit_ensemble(on=on, persist=persist)

    def load(self, path: Union[str, Sequence[str], Sequence[os.PathLike]]):
        """
        Loads up a list of models or a single model from a list of paths or a path,
        implemented in ABC and is called directly from it

        Note that this de-initiates any stored models stored in the Model instance

        Parameters
        ----------
        path:               A str or an Sized or Iterable of str of paths to stored models
        """

        super().load(path=path)

    def evaluate(self, batch_size: int) -> list:
        """
        Evaluates the accuracy of the model, implemented in ABC and is called directly from it

        Parameters
        ----------
        batch_size:          The fractional split of the overall dataset to use for evaluation
        """

        return super().evaluate(batch_size=batch_size)

    def predict(self, to_predict: Union[str, list], interpret: Optional[Callable] = None):
        """
        Predicts the label based on the input string

        Parameters
        ----------
        to_predict:         String to feed into model for predictions
        interpret:          Optional function to interpret predictions
        """

        return super().predict(to_predict=to_predict, interpret=interpret)


class BaseUncasedBERTModel(AbstractBERTModel):
    """A base and uncased version of the BERT model"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bert_vectorizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.bert_data_collator = DataCollatorWithPadding(tokenizer=self.bert_vectorizer)

    def instantiate(self, batch_size: int = 1, shuffle: bool = True, epochs: int = 100):
        """
        This function allows you to convert the input data to

        Parameters
        ----------
        """

        #TODO

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.epochs = epochs
        batch_per_epoch = len(self.model_data.data) // self.batch_size
        total_train_step = int(batch_per_epoch * self.epochs)

        self.model_data.apply(func_map=[lambda x: self.preprocessor(x)], tgt_map=[self.model_data.word_col])
        pass

    def optimise(self, *args, **kwargs):
        """BERT Models do not support explicit optimizations"""

        return super().optimise(*args, **kwargs)

    def fit(self, persist: bool = True):
        """
        Fits the model(s) stored, using the abstract method defined in abstract base class

        Ensemble fitting is done with fit_ensemble() function

        Parameters
        ----------
        persist:                Flag to indicate whether to persist model to disk or not
        """

        pass

    def fit_ensemble(self, *args, **kwargs):
        """BERT Models do not support ensemble models"""

        return super().fit_ensemble(*args, **kwargs)

    def load(self, *args, **kwargs):
        """BERT Models do not support being loaded from disk"""

        return super().load(*args, **kwargs)

    def evaluate(self, batch_size: int) -> list:
        """
        Evaluates the accuracy of the model, implemented in ABC and is called directly from it

        Parameters
        ----------
        batch_size:          The fractional split of the overall dataset to use for evaluation
        """

        pass

    def predict(self, to_predict: Union[str, list], interpret: Optional[Callable] = None):
        """
        Predicts the label based on the input string

        Parameters
        ----------
        to_predict:         String to feed into model for predictions
        interpret:          Optional function to interpret predictions
        """

        pass


class BaseCasedBERTModel(AbstractBaseModel):
    bert_vectorizer: Optional[Any] = None
    bert_data_collator: Optional[DataCollatorWithPadding] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bert_vectorizer = None
        self.bert_data_collator = None


class LargeUncasedBERTModel(AbstractBaseModel):
    bert_vectorizer: Optional[Any] = None
    bert_data_collator: Optional[DataCollatorWithPadding] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bert_vectorizer = None
        self.bert_data_collator = None


class LargeCasedBERTModel(AbstractBaseModel):
    bert_vectorizer: Optional[Any] = None
    bert_data_collator: Optional[DataCollatorWithPadding] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bert_vectorizer = None
        self.bert_data_collator = None


if __name__ == '__main__':
    sm = SimpleModel(ensemble_count=1, model_data=ModelData(path=os.getcwd(), format='csv', on_error='ignore'))
    rnnm = RNNModel(ensemble_count=1, model_data=ModelData(path=os.getcwd(), format='csv', on_error='ignore'))
    birnnm = BidirectionalRNNModel(ensemble_count=1, model_data=ModelData(path=os.getcwd(), format='csv',
                                                                           on_error='ignore'))
