import os

from pydantic import BaseModel, ValidationError, validator
from typing import Union, Iterable, Optional, Tuple, Dict


class Input(BaseModel):

    class Config:
        title = 'ModelInput'
        arbitrary_types_allowed = True
        validate_all = True

    path: Union[os.PathLike, str, Iterable[os.PathLike], Iterable[str]] = ''
    format: str = ''
    on_error: str = ''
    args: Optional[Tuple] = None
    kwargs: Optional[Dict] = None

    @validator('path', always=True)
    def validate_path(cls, v):
        """Validates the path input for type validity and value validity"""

        if isinstance(v, (str, os.PathLike)):
            if os.path.exists(v):
                return v
            else:
                raise FileNotFoundError('Path is invalid')
        elif isinstance(v, Iterable):
            for item in v:
                if not isinstance(item, (str, os.PathLike)):
                    raise TypeError('Iterable does not contain just strings or Pathlike objects')
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


if __name__ == '__main__':
    inputs = Input(path=os.getcwd(), format='csv', on_error='raise', args=(1, ), kwargs={'alternative': 1})
    print(inputs.path)
    print(inputs.format)
    print(inputs.on_error)
    print(inputs.args)
    print(inputs.kwargs)