from numpy import asarray, ndarray
from sqlalchemy import JSON, Float, Integer
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.types import TypeDecorator

from .constants import NP_PRECISION


class EntityBase(DeclarativeBase):
    pass


class FloatArrayType(TypeDecorator):
    """Numpy float vector in memory; JSON list of numbers on the wire."""

    impl = JSON
    cache_ok = True

    def process_bind_param(self, value: ndarray, dialect):
        if value is None:
            return None
        return asarray(value, dtype=NP_PRECISION).tolist()

    def process_result_value(self, value, dialect) -> ndarray:
        return asarray(value, dtype=NP_PRECISION) if value is not None else None


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
