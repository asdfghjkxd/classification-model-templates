import abc
import os
import pandas as pd
import sklearn.model_selection as prep
import logging
import numpy as np
import pickle
import tensorflow as tf

from tensorflow.keras import utils
from sklearn.preprocessing import LabelEncoder
from pydantic import *
from typing import *
from transformers import AutoTokenizer, DataCollatorWithPadding


# +-------------------------------------------------------------------------------------------------------------------+
# |                                     BASE CLASSES FOR BASIC TENSORFLOW MODELS                                      |
# +-------------------------------------------------------------------------------------------------------------------+
class IterableMap(BaseModel):
    """
    This class stores any Iterable of Callables, and two other lists of Iterables containing
    strs of the keys for the underlying pandas DataFrames

    Attributes
    ----------
    func_map:           Sequence/Collection of Callables
    tgt_map:            Sequence/Collection of target columns
    dest_map:           Sequence/Collection of destination columns
    """

    class Config:
        title = 'IterableMap'
        arbitrary_types_allowed = True
        validate_all = True
        smart_union = True
        validate_assignment = True

    df: pd.DataFrame
    func_map: Union[Sequence[Callable]]
    tgt_map: Union[Sequence[Union[int, str]]]
    dest_map: Optional[Sequence[Union[int, str]]] = None

    @validator('df', allow_reuse=True)
    def assert_filled(cls, v):
        if isinstance(v, pd.DataFrame) and not v.empty:
            return v
        else:
            raise TypeError('df must be a filled DataFrame')

    @root_validator(allow_reuse=True)
    def assert_equal_dims(cls, values):
        func, tgt, dest = values.get('func_map'), values.get('tgt_map'), values.get('dest_map')
        if dest is not None:
            if len(func) == len(tgt) == len(dest):
                return values
            else:
                raise AssertionError('Dimensions of func_map, tgt_map, and dest_map must be equal. Offending dim: '
                                     f'{max(map(lambda x: len(x), (func, tgt, dest)))}')
        else:
            if len(func) == len(tgt):
                return values
            else:
                raise AssertionError(f'Dimensions of func_map and tgt_map. Offending dim: '
                                     f'{max(map(lambda x: len(x), (func, tgt)))}')

    @root_validator(allow_reuse=True)
    def assert_valid_col_names(cls, values):
        df, tgt = values.get('df'), values.get('tgt_map')
        for t in tgt:
            if t not in df.columns:
                raise ValueError(f'Cannot find {t} in DataFrame')
        return values

    def map(self) -> pd.DataFrame:
        """Maps the sequence of callables onto the target columns, optionally onto the destination columns"""

        for i in range(len(self.func_map)):
            if self.dest_map is not None:
                self.df[self.dest_map[i]] = self.df[self.tgt_map[i]].apply(self.func_map[i])
            else:
                self.df[self.tgt_map[i]] = self.df[self.tgt_map[i]].apply(self.func_map[i])

        return self.df


class IterableDrop(BaseModel):
    """
    This class helps to map a sequence of column names to lookup on the input pandas DataFrame

    Raises an error when any of the column names are not found in the pandas DataFrame

    All methods that are not validators return the DataFrame after manipulation

    Attributes
    ----------
    df:                     pandas DataFrame to manipulate
    tgt_map:                A sequence of column names to drop
    axis:                   An integer taking on values 0 or 1 strictly
    """

    class Config:
        title = 'IterableDrop'
        arbitrary_types_allowed = True
        allow_mutation = True
        smart_union = True
        validate_assignment = True

    df: pd.DataFrame
    tgt_map: Union[Sequence[Union[int, str]]]
    axis: conint(ge=0, le=1)

    @root_validator(allow_reuse=True)
    def assert_valid_col_names(cls, values):
        df, col_names = values.get('df'), values.get('tgt_map')
        for col in col_names:
            if col not in df.columns:
                raise ValueError(f'Cannot find {col} in DataFrame columns')
        return values

    def drop(self) -> pd.DataFrame:
        """Drops columns from input DataFrame inplace"""

        for it in self.tgt_map:
            self.df.drop(labels=it, axis=self.axis, inplace=True)
        return self.df


# +-------------------------------------------------------------------------------------------------------------------+
# |                                     ABSTRACT BASE CLASS FOR ALL MODEL CLASSES                                     |
# +-------------------------------------------------------------------------------------------------------------------+
class AbstractModelInput(BaseModel, abc.ABC):
    """
    This is an abstract class to define model inputs

    Attributes
    ----------
    path:                   A os.PathLike, str or sequence of two aforementioned types
    format:                 A file format string
    on_error:               Behaviour when an operation fails
    args:                   A tuple/*args of positional arguments [Optional]
    kwargs:                 A dict/**kwargs of keyword arguments [Optional]

    > Optional Variables [not necessary to define as it will be updated in the class when operations are performed]
    X:                      A numpy array containing features of the dataset
    y:                      A numpy array containing the actual result
    X_train:                A numpy array containing the set of features to train
    X_test:                 A numpy array containing the set of features to test
    y_train:                A numpy array containing the result of the training features
    y_test:                 A numpy array containing the result of the testing features
    encoder:                A LabelEncoder class for encoding feature labels (unfitted)
    train_test_split:       A ratio of train-test split for dataset
    """

    class Config:
        title = 'ModelInput'
        arbitrary_types_allowed = True
        allow_mutation = True
        smart_union = True
        validate_assignment = True

    path: Union[os.PathLike, str, Union[Sequence[os.PathLike], Sequence[str]]]
    format: str
    on_error: str = 'raise'
    args: Optional[Tuple] = ()
    kwargs: Optional[Dict] = {}
    data: Optional[pd.DataFrame]
    word_col: Optional[str]
    label_col: Optional[str]
    X: Optional[Any]
    y: Optional[Any]
    X_train: Optional[Any]
    X_test: Optional[Any]
    y_train: Optional[Any]
    y_test: Optional[Any]
    encoder: Optional[LabelEncoder]
    train_test_split: confloat(gt=0., lt=1.) = 0.8

    @validator('path', allow_reuse=True)
    def validate_path(cls, v):
        """Validates the path input for type validity and value validity"""

        if isinstance(v, (str, os.PathLike)):
            if os.path.exists(v):
                return v
            else:
                raise FileNotFoundError('Path is invalid')
        elif isinstance(v, Sequence):
            for item in v:
                if not isinstance(item, (str, os.PathLike)):
                    raise TypeError('Sequence does not contain just strings or Pathlike objects')
                else:
                    if not os.path.exists(item):
                        raise FileNotFoundError('Path is invalid')
            return v
        else:
            raise TypeError('Path is not Pathlike object or str')

    @validator('format', allow_reuse=True)
    def validate_format(cls, v):
        """Validates the input file format against a list of predetermined strings"""

        if v in ['csv', 'xlsx', 'json']:
            return v
        else:
            raise ValueError(f'Format {v} is not recognised')

    @validator('on_error', allow_reuse=True)
    def validate_error(cls, v):
        """Validates on_error argument against a list of predetermined strings"""

        if v in ['ignore', 'default', 'raise']:
            return v
        else:
            raise ValueError(f'Format {v} is not recognised')

    @abc.abstractmethod
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def read(self):
        """
        Method which reads the files and saves it into the class attributes

        If arbitrary arguments are provided, they are passed directly into
        the pandas read_*() function; any exception that occurs from incorrect
        arguments will be raised

        If on_error is specified and that *args and **kwargs are specified, the errors
        raised by the read_*() functions will be processed as such
        """

        def _read(curr_path: Union[str, os.PathLike]) -> pd.DataFrame:
            """
            Internal pd.read_* functions

            Parameters
            ----------
            curr_path:      str or Pathlike object that references a dataset file
            """

            try:
                if self.args is not None or self.kwargs is not None:
                    if len(self.args) > 0 or len(self.kwargs) > 0:
                        if self.format == 'csv':
                            return pd.read_csv(curr_path, *self.args, **self.kwargs).astype(str)
                        elif self.format == 'xlsx':
                            return pd.read_excel(curr_path, *self.args, **self.kwargs).astype(str)
                        elif self.format == 'json':
                            return pd.read_json(curr_path, *self.args, **self.kwargs).astype(str)
                    else:
                        if self.format == 'csv':
                            return pd.read_csv(curr_path).astype(str)
                        elif self.format == 'xlsx':
                            return pd.read_excel(io=curr_path).astype(str)
                        elif self.format == 'json':
                            return pd.read_json(curr_path).astype(str)
                else:
                    if self.format == 'csv':
                        return pd.read_csv(curr_path).astype(str)
                    elif self.format == 'xlsx':
                        return pd.read_excel(io=curr_path).astype(str)
                    elif self.format == 'json':
                        return pd.read_json(curr_path).astype(str)
            except Exception as exc:
                raise exc

        try:
            if isinstance(self.path, (str, os.PathLike)):
                self.data = _read(curr_path=self.path)
            elif isinstance(self.path, Sequence):
                temp = [_read(curr_path=p) for p in self.path]
                self.data = pd.concat(temp, axis=0)
        except Exception as ex:
            if self.on_error == 'ignore':
                logging.warning('Dataset is passed as an Exception was encountered while processing the dataset')
            elif self.on_error == 'default':
                # silence exception and loggers, and do nothing
                pass
            elif self.on_error == 'raise':
                raise ex

    @abc.abstractmethod
    def preprocess(self, *args, **kwargs):
        raise NotImplementedError

    def apply(self, func_map: Union[Sequence[Callable]], tgt_map: Union[Sequence[Union[int, str]]],
              dest_map: Optional[Sequence[Union[int, str]]] = None) -> None:
        """
        Applies a function map onto a sequence of column names, and optionally output the returns to another
        sequence of column names
        """

        assert isinstance(self.data, pd.DataFrame) and not self.data.empty, 'DataFrame cannot be empty'

        itermap = IterableMap(df=self.data, func_map=func_map, tgt_map=tgt_map, dest_map=dest_map)
        self.data = itermap.map()

    def drop(self, tgt_map: Union[Sequence[Union[int, str]]], axis: StrictInt = 0) -> None:
        """Drops the columns in the DataFrame specified in the target map input"""

        assert isinstance(self.data, pd.DataFrame) and not self.data.empty, 'DataFrame cannot be empty'

        iterdrop = IterableDrop(df=self.data, tgt_map=tgt_map, axis=axis)
        self.data = iterdrop.drop()

    def shape(self) -> tuple:
        """Returns the shape of the DataFrame stored"""

        assert isinstance(self.data, pd.DataFrame) and not self.data.empty, 'DataFrame cannot be empty'

        return self.data.shape if self.data is not None else (0,)

    @abc.abstractmethod
    def is_processed(self) -> bool:
        """Simple check to see if data is properly split and processed"""

        def _check(_in: Any):
            """Internal check if x is of pd.DataFrame or np.ndarray"""

            if isinstance(_in, pd.DataFrame):
                return not _in.empty
            elif isinstance(_in, np.ndarray):
                return _in.size != 0
            else:
                return False if _in is None else True

        return all(list(map(lambda x: _check(x),
                            (self.data, self.X, self.y, self.X_train, self.X_test, self.y_train, self.y_test))))


class ModelData(AbstractModelInput):
    """Base Model Data Class for non-BERT based models"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.read()

    def preprocess(self, word_col: str, label_col: str, train_test_split: StrictFloat):
        """
        Preprocesses the dataset and splits the dataset into train-test sets

        Extracts out the words column and label column and processes them

        Vectorisation of the dataset will be done during the instantiation of the model,
        the vectorisation layer is not instantiated here
        """

        assert isinstance(self.data, pd.DataFrame) and not self.data.empty, 'DataFrame cannot be empty'

        self.train_test_split = train_test_split

        try:
            # shuffle the dataset
            self.data = self.data.sample(frac=1).reset_index(drop=True)

            # split by labels
            to_process = self.data[[word_col, label_col]]
            y = to_process[label_col]
            X = to_process[word_col].to_numpy()
        except KeyError:
            raise ValueError('Invalid word_col or label_col argument')
        except Exception as ex:
            raise ex
        else:
            # set states for word and label cols
            self.word_col = word_col
            self.label_col = label_col

            # turn the labels into dummy labels
            self.encoder = LabelEncoder()
            self.encoder.fit(y)
            self.X = X
            self.y = utils.to_categorical(self.encoder.transform(y))

            # persist encoder
            with open(f'models/encoders/encoder_model.pkl', 'wb') as f:
                pickle.dump(self.encoder, f)

            # split by train-test
            try:
                self.X_train, self.X_test, self.y_train, self.y_test = prep.train_test_split(
                    self.X, self.y, train_size=self.train_test_split, random_state=420
                )
            except Exception as ex:
                raise ex

    def is_processed(self) -> bool:
        """Simple check to see if data is properly split and processed"""

        return super().is_processed()


class BERTModelData(AbstractModelInput):
    """BERT Model Data"""

    vectorizer: Any = None
    collator: Optional[DataCollatorWithPadding] = None
    train_dataset: Optional[tf.data.Dataset] = None
    test_dataset: Optional[tf.data.Dataset] = None
    batch_size: Optional[int] = 1

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(kwargs.get('load_from_name'))
            self.collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
            self.train_dataset = None
            self.test_dataset = None
            self.batch_size = 1
        except Exception as ex:
            raise ex
        else:
            self.read()

    @validator('batch_size', allow_reuse=True)
    def assert_valid_batches(cls, v):
        if isinstance(v, int):
            if v > 0:
                return v
            else:
                raise ValueError('Batch size cannot be zero or negative')
        else:
            raise TypeError('Batch Size must be an int')

    def preprocess(self, word_col: str, label_col: str, train_test_split: StrictFloat,
                   batch_size: int = 1):
        """
        Preprocesses the dataset and splits the dataset into train-test sets

        Extracts out the words column and label column and processes them

        Vectorisation of the dataset will be done during the instantiation of the model,
        the vectorisation layer is not instantiated here
        """

        self.train_test_split = train_test_split
        self.batch_size = batch_size
        assert isinstance(self.data, pd.DataFrame) and not self.data.empty, 'DataFrame cannot be empty'

        try:
            # shuffle the dataset
            self.data = self.data.sample(frac=1).reset_index(drop=True)

            # split by labels
            to_process = self.data[[word_col, label_col]]
            y = to_process[label_col]
            X = to_process[word_col].to_numpy()
        except KeyError:
            raise ValueError('Invalid word_col or label_col argument')
        except Exception as ex:
            raise ex
        else:
            # set states for word and label cols
            self.word_col = word_col
            self.label_col = label_col

            # turn the labels into dummy labels
            self.encoder = LabelEncoder()
            self.encoder.fit(y)
            self.X = X
            self.y = utils.to_categorical(self.encoder.transform(y))

            # persist encoder
            self.tokenizer.save_pretrained('../../models/tokenizer')

            # split by train-test
            try:
                self.X_train, self.X_test, self.y_train, self.y_test = prep.train_test_split(
                    self.X, self.y, train_size=self.train_test_split, random_state=420
                )
            except Exception as ex:
                raise ex
            else:
                self.X_train = self.tokenizer(self.X_train, truncation=True, padding=True)
                self.X_test = self.tokenizer(self.X_test, truncation=True, padding=True)
                self.train_dataset = tf.data.Dataset.from_tensor_slices((
                    self.X_train, self.y_train
                )).batch(batch_size=self.batch_size)
                self.test_dataset = tf.data.Dataset.from_tensor_slices((
                    self.X_test, self.y_test
                )).batch(batch_size=self.batch_size)

    def is_processed(self) -> bool:
        """Simple check to see if data is properly split and processed"""

        return super().is_processed()
