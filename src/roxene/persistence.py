import numpy as np
import pickle
import sqlalchemy.types
import torch
from sqlalchemy import PickleType
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.ext.mutable import Mutable

from .constants import TORCH_PRECISION


class EntityBase(DeclarativeBase):
    pass


class TrackedTensor(Mutable):

    variable: torch.Tensor

    def __init__(self, variable: torch.Tensor):
        super(Mutable, self).__init__()
        self.variable = variable

    @classmethod
    def coerce(cls, key, value):
        if not isinstance(value, TrackedTensor):
            if isinstance(value, torch.Tensor):
                return TrackedTensor(value)
            return Mutable.coerce(key, value)
        return value

    def __getattr__(self, item):
        return getattr(self.variable, item)

    def __getitem__(self, key):
        return self.variable[key]

    def __setitem__(self, key, value):
        # Convert numpy values to tensors if needed
        if isinstance(value, (np.ndarray, np.generic)):
            value = torch.tensor(value, dtype=self.variable.dtype)
        self.variable[key] = value
        self.changed()

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        # Unwrap TrackedVariable instances to their underlying tensors
        def unwrap(x):
            return x.variable if isinstance(x, TrackedTensor) else x
        args = tuple(unwrap(a) if not isinstance(a, (tuple, list)) else 
                     type(a)(unwrap(i) for i in a) for a in args)
        kwargs = {k: unwrap(v) for k, v in kwargs.items()}
        return func(*args, **kwargs)

    def __eq__(self, other):
        if isinstance(other, torch.Tensor):
            return torch.equal(self.variable, other)
        return super(Mutable, self).__eq__(other)

    @property
    def shape(self):
        return self.variable.shape


class WrappedVariable(sqlalchemy.types.TypeDecorator):

    impl = PickleType
    cache_ok = True

    def process_bind_param(self, value, dialect) -> np.ndarray:
        return value.variable.numpy()

    def process_result_value(self, value: np.ndarray, dialect):
        return TrackedTensor(torch.tensor(value, dtype=TORCH_PRECISION)) if value is not None else None


class WrappedTensor(sqlalchemy.types.TypeDecorator):

    impl = sqlalchemy.types.LargeBinary
    cache_ok = True

    def process_bind_param(self, value: torch.Tensor, dialect) -> bytes:
        return pickle.dumps(value.numpy(), protocol=5)

    def process_result_value(self, value: bytes, dialect) -> torch.Tensor:
        return torch.tensor(pickle.loads(value), dtype=TORCH_PRECISION) if value else None
