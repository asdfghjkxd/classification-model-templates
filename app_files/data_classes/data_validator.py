import os
import pandas as pd
import logging

from enum import IntEnum
from pydantic import BaseModel, ValidationError, validator, root_validator, StrictInt
from typing import Union, Iterable, Optional, Tuple, Dict, Sequence, Callable, Any


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

    def drop(self) -> pd.DataFrame:
        """Drops columns from input DataFrame"""

        for it in self.tgt_map:
            self.df.drop(labels=it, axis=axis, inplace=True)

        return self.df


class Input(BaseModel):
    """This class allows for the verification and manipulation of input data for the model"""

    class Config:
        title = 'ModelInput'
        arbitrary_types_allowed = True
        allow_mutation = True
        smart_union = True

    path: Union[os.PathLike, str, Union[Sequence[os.PathLike], Sequence[str]]] = ''
    format: str
    on_error: str = 'raise'
    args: Optional[Tuple] = None
    kwargs: Optional[Dict] = None
    data: Any = None
    X: Any = None
    y: Any = None
    X_train: Any = None
    X_test: Any = None
    y_train: Any = None
    y_test: Any = None
    encoder: Any = None
    train_test_split: Any = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.encoder = None
        self.train_test_split = None
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

    @validator('args', always=True)
    def validate_args(cls, v):
        """Validates the sequence of positional argument inputs"""

        if isinstance(v, (tuple, type(None))):
            return v
        else:
            raise TypeError('args is not a series of positional arguments')

    @validator('kwargs', always=True)
    def validate_kwargs(cls, v):
        """Validates the sequence of keyword argument inputs"""

        if isinstance(v, (dict, type(None))):
            return v
        else:
            raise TypeError('kwargs is not a series of keyword arguments')

    def read(self) -> None:
        """
        Method which reads the files and saves it into the class attributes

        If arbitrary arguments are provided, they are passed directly into
        the pandas read_*() function; any exception that occurs from incorrect
        arguments will be raised

        If on_error is specified and that *args and **kwargs are specified, the errors
        raised by the read_*() functions will be processed as such
        """

        def _read(self, curr_path: Union[str, os.PathLike]) -> pd.DataFrame:
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
                        return pd.read_excel(curr_path).astype(str)
                    elif self.format == 'json':
                        return pd.read_json(curr_path).astype(str)
            except Exception as exc:
                raise exc

        try:
            if isinstance(self.path, (str, os.PathLike)):
                self.data = _read(self, curr_path=self.path)
            elif isinstance(self.path, Sequence):
                temp = [_read(self, curr_path=p) for p in self.path]
                self.data = pd.concat(temp, axis=0)
        except Exception as ex:
            if self.on_error == 'ignore':
                logging.warning('Dataset is passed as an Exception was encountered while processing the dataset')
            elif self.on_error == 'default':
                # silence exception and default to nothing
                pass
            elif self.on_error == 'raise':
                raise ex

    def apply(self, func_map: Union[Sequence[Callable]], tgt_map: Union[Sequence[Union[int, str]]],
              dest_map: Optional[Sequence[Union[int, str]]] = None) -> None:
        """
        Applies a function map onto a sequence of column names, and optionally output the returns to another
        sequence of column names
        """

        itermap = IterableMap(df=self.data, func_map=func_map, tgt_map=tgt_map, dest_map=dest_map)
        self.data = itermap.map()

    def drop(self, tgt_map: Union[Sequence[Union[int, str]]], axis: StrictInt = 0) -> None:
        """Drops the columns in the DataFrame specified in the target map input"""

        iterdrop = IterableDrop(df=self.data, tgt_map=tgt_map, axis=axis)
        self.data = iterdrop.drop()

    def shape(self) -> tuple:
        """Returns the shape of the DataFrame stored"""

        return self.data.shape if self.data is not None else (0,)

    def is_processed(self) -> bool:
        """Simple check to see if data is properly split and processed"""

        def _check(inputs):
            """Internal check if x is of pd.DataFrame or np.ndarray"""

            if isinstance(inputs, pd.DataFrame):
                return not inputs.empty
            elif isinstance(inputs, np.ndarray):
                return inputs.size != 0
            else:
                return False if None else True

        return all(list(map(lambda x: _check(x),
                            (self.data, self.X, self.y, self.X_train, self.X_test, self.y_train, self.y_test))))


if __name__ == '__main__':
    inputs = Input(path=os.getcwd(), format='csv', on_error='ignore', args=(1, ), kwargs={'1': 2})
    print(inputs.path)
    print(inputs.format)
    print(inputs.on_error)
    print(inputs.args)
    print(inputs.kwargs)

    iterables = IterableMap(df=pd.DataFrame(data=[0, 1, 2]),
                            func_map=[lambda x: x + 1], tgt_map=[0], dest_map=[1])
    print(iterables.func_map)
    print(iterables.tgt_map)
    print(iterables.dest_map)
    print(iterables.df)
    iterables.map()
    print(iterables.df)
