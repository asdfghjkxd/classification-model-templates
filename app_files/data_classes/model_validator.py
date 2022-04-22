import os

from pydantic import BaseModel, ValidationError, validator
from typing import Union, Iterable, Optional, Tuple, Dict


class ModelConfig(BaseModel):

    class Config:
        title = 'ModelConfig'
        arbitrary_types_allowed = True
        validate_all = True
        allow_mutation = True

