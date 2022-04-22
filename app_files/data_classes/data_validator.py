import os
import pandas as pd
import sklearn.model_selection as prep
import logging
import numpy as np
import pickle

from sklearn.preprocessing import LabelEncoder
from pydantic import BaseModel, ValidationError, validator, root_validator, StrictInt, StrictFloat
from typing import *


class IterableMap(BaseModel):
    """
    This class stores any Iterable of Callables, and two other lists of Iterables containing
    strs of the keys for the underlying pandas DataFrames

    func_map:           Sequence/Collection of Callables
    tgt_map:            Sequence/Collection of target columns
    dest_map:           Sequence/Collection of destination columns
    """

    df: pd.DataFrame
    func_map: Union[Sequence[Callable]]
    tgt_map: Union[Sequence[Union[int, str]]]
    dest_map: Optional[Sequence[Union[int, str]]] = None

    class Config:
        title = 'IterableMap'
        arbitrary_types_allowed = True
        validate_all = True
        smart_union = True

    @validator('df', always=True)
    def assert_filled(cls, v):
        if isinstance(v, pd.DataFrame) and not v.empty:
            return v
        else:
            raise TypeError('df must be a filled DataFrame')

    @root_validator
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

    def map(self) -> pd.DataFrame:
        """Maps the sequence of callables onto the target columns, optionally onto the destination columns"""

        for i in range(len(self.func_map)):
            try:
                if self.dest_map is not None:
                    self.df[self.dest_map[i]] = self.df[self.tgt_map[i]].apply(self.func_map[i])
                else:
                    self.df[self.tgt_map[i]] = self.df[self.tgt_map[i]].apply(self.func_map[i])
            except KeyError:
                raise ValueError(f'Key {self.tgt_map[i]} not found in DataFrame')

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

    df: pd.DataFrame
    tgt_map: Union[Sequence[Union[int, str]]]
    axis: StrictInt

    @validator('axis', always=True)
    def assert_normalized(cls, v):
        if v == 0 or v == 1:
            return v
        else:
            raise ValueError('axis argument must be 0 or 1')

    @root_validator
    def assert_valid_col_names(cls, values):
        df, col_names = values.get('df'), values.get('tgt_map')

        for col in col_names:
            if col not in df.columns:
                raise ValueError(f'{col} not found in DataFrame')

        return values

    def drop(self) -> pd.DataFrame:
        """Drops columns from input DataFrame"""

        for it in self.tgt_map:
            self.df.drop(labels=it, axis=axis, inplace=True)

        return self.df


class Input(BaseModel):
    """
    This class allows for the verification and manipulation of input data for the model

    Attributes
    ----------
    path:                   A os.PathLike, str or sequence of two aforementioned types
    format:                 A file format string
    on_error:               Behaviour when an operation fails
    args:                   A tuple/*args of positional arguments
    kwargs:                 A dict/**kwargs of keyword arguments
    X:                      A numpy array
    y:                      A numpy array
    X_train:                A numpy array
    X_test:                 A numpy array
    y_train:                A numpy array
    y_test:                 A numpy array
    encoder:                A LabelEncoder class
    train_test_split:       A ratio of train-test split
    """

    class Config:
        title = 'ModelInput'
        arbitrary_types_allowed = True
        allow_mutation = True
        smart_union = True

    path: Union[os.PathLike, str, Union[Sequence[os.PathLike], Sequence[str]]]
    format: str
    on_error: str = 'raise'
    args: Optional[Tuple] = None
    kwargs: Optional[Dict] = None
    data: Optional[pd.DataFrame]
    X: Any = None
    y: Any = None
    X_train: Any = None
    X_test: Any = None
    y_train: Any = None
    y_test: Any = None
    encoder: Optional[LabelEncoder]
    train_test_split: StrictFloat = 0.8

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.encoder = None
        self.read()

    @validator('path', always=True)
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

    @validator('format', always=True)
    def validate_format(cls, v):
        """Validates the input file format against a list of predetermined strings"""

        if isinstance(v, str):
            if v in ['csv', 'xlsx', 'json']:
                return v
            else:
                raise ValueError(f'Format {v} is not recognised')
        else:
            raise TypeError(f'File format {v} is not of type str')

    @validator('on_error', always=True)
    def validate_error(cls, v):
        """Validates on_error argument against a list of predetermined strings"""

        if isinstance(v, str):
            if v in ['ignore', 'default', 'raise']:
                return v
            else:
                raise ValueError(f'Format {v} is not recognised')
        else:
            raise TypeError(f'File format {v} is not of type str')

    @validator('train_test_split', always=True)
    def validate_train_test(cls, v):
        """Validates the train_test_split variable to be between 0 to 1"""

        if 0 < v < 1:
            return v
        else:
            raise ValueError('train_test_split must be between 0 to 1 (not inclusive)')

    def read(self) -> None:
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
                # silence exception and default to nothing
                pass
            elif self.on_error == 'raise':
                raise ex

    def preprocess(self, word_col: str, label_col: str, train_test_split: StrictFloat):
        """
        Preprocesses the dataset and splits the dataset into train-test sets

        Extracts out the words column and label column and processes them

        Vectorisation of the dataset will be done during the instantiation of the model,
        the vectorisation layer is not instantiated here
        """

        assert isinstance(self.data, pd.DataFrame) and not self.data.empty

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

    def apply(self, func_map: Union[Sequence[Callable]], tgt_map: Union[Sequence[Union[int, str]]],
              dest_map: Optional[Sequence[Union[int, str]]] = None) -> None:
        """
        Applies a function map onto a sequence of column names, and optionally output the returns to another
        sequence of column names
        """

        assert isinstance(self.data, pd.DataFrame) and not self.data.empty

        itermap = IterableMap(df=self.data, func_map=func_map, tgt_map=tgt_map, dest_map=dest_map)
        self.data = itermap.map()

    def drop(self, tgt_map: Union[Sequence[Union[int, str]]], axis: StrictInt = 0) -> None:
        """Drops the columns in the DataFrame specified in the target map input"""

        assert isinstance(self.data, pd.DataFrame) and not self.data.empty

        iterdrop = IterableDrop(df=self.data, tgt_map=tgt_map, axis=axis)
        self.data = iterdrop.drop()

    def shape(self) -> tuple:
        """Returns the shape of the DataFrame stored"""

        assert isinstance(self.data, pd.DataFrame) and not self.data.empty

        return self.data.shape if self.data is not None else (0,)

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


if __name__ == '__main__':
    inputs = Input(path=os.getcwd(), format='csv', on_error='ignore', args=(1, ), kwargs={'1': 2}, new='pls')
    print(inputs.path)
    print(inputs.format)
    print(inputs.on_error)
    print(inputs.args)
    print(inputs.kwargs)
    print(inputs.on_error)
    inputs.read()
    # inputs.preprocess(word_col='0', label_col='1', train_test_split=0.8)

    iterables = IterableMap(df=pd.DataFrame(data=[0, 1, 2]),
                            func_map=[lambda x: x + 1], tgt_map=[0], dest_map=[1])
    print(iterables.func_map)
    print(iterables.tgt_map)
    print(iterables.dest_map)
    print(iterables.df)
    iterables.map()
    print(iterables.df)
