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


class TrackedVariable(Mutable):

    variable: torch.Tensor

    def __init__(self, variable: torch.Tensor):
        super(Mutable, self).__init__()
        self.variable = variable

    @classmethod
    def coerce(cls, key, value):
        if not isinstance(value, TrackedVariable):
            if isinstance(value, torch.Tensor):
                return TrackedVariable(value)
            return Mutable.coerce(key, value)
        return value

    def assign(self, value):
        self.variable.data.copy_(value)
        self.changed()

    def __getattr__(self, item):
        return getattr(self.variable, item)

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
        # Handle both TrackedVariable and plain torch.Tensor
        if isinstance(value, TrackedVariable):
            return value.variable.detach().cpu().numpy()
        elif isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return value

    def process_result_value(self, value: np.ndarray, dialect):
        # Return TrackedVariable directly so it's always the right type
        if value is not None:
            return TrackedVariable(torch.tensor(value, dtype=TORCH_PRECISION))
        return None


class WrappedTensor(sqlalchemy.types.TypeDecorator):

    impl = sqlalchemy.types.LargeBinary
    cache_ok = True

    def process_bind_param(self, value: torch.Tensor, dialect) -> bytes:
        return pickle.dumps(value.detach().cpu().numpy(), protocol=5)

    def process_result_value(self, value: bytes, dialect) -> torch.Tensor:
        return torch.tensor(pickle.loads(value), dtype=TORCH_PRECISION) if value else None
