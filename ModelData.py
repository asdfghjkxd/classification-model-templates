import toml
import os
import pandas as pd
import sklearn.model_selection as prep
import logging

from typing import *
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import utils


class ModelData:
    """
    Preprocesses the dataset to be used to feed to the ML model

    All data to be used for the training must be stored and obtained from this class
    """

    def __init__(self, path: Union[str, os.PathLike, Iterable[Union[str, os.PathLike]]],
                 file_format: str = 'csv', on_error: Optional[str] = 'raise',
                 *args, **kwargs) -> None:
        """
        Initialises the ModelData class by reading the one dataset first

        To use more than one dataset, store the filepaths in an Iterable and pass
        the variable into the path argument above

        Parameters
        ----------
        path:           str or Pathlike or Iterable containing str or Pathlike
                        object that references a dataset file
        file_format:    str, that must be 'csv', 'xlsx' or 'json'
        on_error:       str, that must be 'ignore', 'default' or 'raise'
        """

        self.path = self._validate(path, (str, os.PathLike, Iterable))
        self.file_format = self._validate(file_format, str, ('csv', 'xlsx', 'json'))
        self.on_error = self._validate(on_error, str, ('ignore', 'default', 'raise'))
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.encoder = None
        self.train_test_split = None
        self.read(path=self.path, file_format=self.file_format, on_error=self.on_error,
                  *args, **kwargs)

    def __call__(self) -> pd.DataFrame:
        """Returns the stored dataframe when the class object is called"""

        return self.data if self.data is not None else None

    def read(self, path: Union[str, os.PathLike, Iterable[Union[str, os.PathLike]]],
             file_format: str = 'csv', on_error: Optional[str] = 'raise',
             *args, **kwargs) -> None:
        """
        Method which reads the files and saves it into the class attributes

        If arbitrary arguments are provided, they are passed directly into
        the pandas read_*() function; any exception that occurs from incorrect
        arguments will be raised

        If on_error is specified and that *args and **kwargs are specified, the errors
        raised by the read_*() functions will be processed as such

        Parameters
        ----------
        path:           str or Pathlike or Iterable containing str or Pathlike
                        object that references a dataset file
        file_format:    str, that must be 'csv', 'xlsx' or 'json'
        on_error:       str, that must be 'ignore', 'default' or 'raise'
        """

        def _read(_path: Union[str, os.PathLike, Iterable[Union[str, os.PathLike]]],
                  _file_format: str = 'csv', *_args, **_kwargs) -> pd.DataFrame:
            """
            Internal pd.read_* functions

            Parameters
            ----------
            _path:              str or Pathlike or Iterable containing str or Pathlike
                                object that references a dataset file
            _file_format:       str, that must be 'csv', 'xlsx' or 'json'
            """

            if len(_args) > 0 or len(_kwargs) > 0:
                if _file_format == 'csv':
                    df = pd.read_csv(_path, *_args, **_kwargs).astype(str)
                elif _file_format == 'xlsx':
                    df = pd.read_excel(_path, *_args, **_kwargs).astype(str)
                elif _file_format == 'json':
                    df = pd.read_json(_path, *_args, **_kwargs).astype(str)
            else:
                if _file_format == 'csv':
                    df = pd.read_csv(_path).astype(str)
                elif _file_format == 'xlsx':
                    df = pd.read_excel(_path).astype(str)
                elif _file_format == 'json':
                    df = pd.read_json(_path).astype(str)

            return df

        try:
            if isinstance(path, str):
                self.data = _read(_path=path, _file_format=file_format, *args, **kwargs)
            else:
                temp = [_read(_path=p, _file_format=file_format, *args, **kwargs) for
                        p in path]
                self.data = pd.concat(temp, axis=0)
        except Exception as ex:
            if on_error == 'ignore':
                logging.warning('Dataset is passed as an Exception was encountered '
                                'while processsing the dataset')
            elif on_error == 'default':
                self.data = _read(_path=path, _file_format=file_format)
            elif on_error == 'raise':
                raise ex

    def apply(self, functions: Iterable[Callable] or Sized,
              target: Iterable[str] or Sized,
              destination: Iterable[str] or Sized[str]) -> None:
        """
        Wrapper function around pd.apply() to sequential modification of pandas
        DataFrames

        Arguments
        ---------
        functions:          Any Iterable datatypes that implements .__len__() or can be iterated,
                            represents the list of functions to apply to frame
        target:             Any Iterable datatypes that implements .__len__() or can be iterated,
                            represents the target columns to apply functions to
        destination:        Any Iterable datatypes that implements .__len__() or can be iterated,
                            represents the destination columns to store data in, optional as
                            target == destination by default

        Raises
        ------
        KeyError:           If target or destination contains erroneous column name values
        """

        if len(destination) > 0:
            assert len(functions) == len(target) == len(destination)

            function_call_pair = list(zip(functions, target, destination))
            for (func, targ, dest) in function_call_pair:
                try:
                    self.data[dest] = self.data[targ].apply(func)
                except KeyError:
                    raise ValueError(f'Key {targ} or {dest} is not valid')
                except Exception:
                    raise
        else:
            assert len(functions) == len(target)

            function_call_pair = list(zip(functions, target))
            for (func, row) in function_call_pair:
                try:
                    self.data[row] = self.data[row].apply(func)
                except KeyError:
                    raise ValueError(f'Key {row} is not valid')
                except Exception:
                    raise

    def drop(self, cols: Union[Iterable[str], str]):
        """
        Drops the passed in col names from the dataset

        Parameters
        ----------
        cols:               An Iterable of strings or string that contains columns to drop
                            from dataset
        """

        cols = self._validate(cols, (Iterable, str))

        if isinstance(cols, Iterable) and not isinstance(cols, str):
            for col in cols:
                try:
                    self.data.drop([col], axis=1, inplace=True)
                except KeyError:
                    raise ValueError('cols contains column headers that are not valid')
        else:
            try:
                self.data.drop([cols], axis=1, inplace=True)
            except KeyError:
                raise ValueError('cols contains column headers that are not valid')

    def preprocess(self, train_test_split: float, word_col: str, label_col: str) -> None:
        """
        Preprocesses the dataset and splits the dataset into train-test sets

        Extracts out the words column and label column and processes them

        Vectorisation of the dataset will be done during the instantiation of the model,
        the vectorisation layer is not instantiated here

        Parameters
        ----------
        train_test_split:        A float to denote the fractional split for dataset
        word_col:                Column containing words to vectorise
        label_col:               Column containing labels
        """

        # validate the ratio first
        self.train_test_split = self._validate(train_test_split, float)
        if not 0 < self.train_test_split < 1:
            self.train_test_split = None
            raise ValueError('train_test_split parameter cannot be less than or equal to '
                             '0 or more than or equal to 1')

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

            # split by train-test
            try:
                self.X_train, self.X_test, self.y_train, self.y_test = prep.train_test_split(
                    self.X, self.y, train_size=self.train_test_split, random_state=420
                )
            except Exception as ex:
                raise ex

    def shape(self) -> tuple:
        """Returns the shape of the DataFrame stored"""

        return self.data.shape if self.data is not None else (0,)

    def is_processed(self) -> bool:
        """Simple check to see if data is properly split and processed"""

        for var in (self.data, self.X, self.y, self.X_train, self.X_test, self.y_train, self.y_test):
            if var is None:
                return False

        return True

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