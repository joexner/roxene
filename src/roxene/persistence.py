import torch
from sqlalchemy import PickleType, Float, Integer
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.types import TypeDecorator

from .constants import DEVICE


class EntityBase(DeclarativeBase):
    pass


class WrappedTensor(TypeDecorator):
    impl = PickleType
    cache_ok = True

    def process_bind_param(self, value, dialect) -> torch.Tensor:
        if value is None:
            return None
        t = value.tensor if hasattr(value, "tensor") else value
        return t.detach().cpu()

    def process_result_value(self, value: torch.Tensor, dialect):
        return value.to(DEVICE) if value is not None else None


class NumpySafeFloat(TypeDecorator):
    impl = Float
    cache_ok = True

    def process_bind_param(self, value, dialect):
        return float(value) if value is not None else None


class NumpySafeInt(TypeDecorator):
    impl = Integer
    cache_ok = True

    def process_bind_param(self, value, dialect):
        return int(value) if value is not None else None
