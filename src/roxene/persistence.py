import numpy as np
import pickle
import sqlalchemy.types
import tensorflow as tf
from sqlalchemy import PickleType
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.ext.mutable import Mutable

from .constants import TF_PRECISION


class EntityBase(DeclarativeBase):
    pass


class TrackedVariable(Mutable):

    variable: tf.Variable

    def __init__(self, variable: tf.Variable):
        super(Mutable, self).__init__()
        self.variable = variable

    @classmethod
    def coerce(cls, key, value):
        if not isinstance(value, TrackedVariable):
            if isinstance(value, tf.Variable):
                return TrackedVariable(value)
            return Mutable.coerce(key, value)
        return value

    def assign(self, value):
        self.variable.assign(value)
        self.changed()

    def __getattr__(self, item):
        return getattr(self.variable, item)

    def __eq__(self, other):
        if isinstance(other, tf.Variable):
            return super(tf.Variable, self).__eq__(other)
        return super(Mutable, self).__eq__(other)

    @property
    def shape(self):
        return self.variable.shape


class WrappedVariable(sqlalchemy.types.TypeDecorator):

    impl = PickleType

    def process_bind_param(self, value: tf.Variable, dialect) -> np.ndarray:
        return value.numpy()

    def process_result_value(self, value: np.ndarray, dialect) -> tf.Variable:
        return tf.Variable(initial_value=value, dtype=TF_PRECISION) if (value is not None) else None


class WrappedTensor(sqlalchemy.types.TypeDecorator):

    impl = sqlalchemy.types.BLOB

    def process_bind_param(self, value: tf.Tensor, dialect) -> bytes:
        return pickle.dumps(value.numpy(), protocol=5)

    def process_result_value(self, value: bytes, dialect) -> tf.Tensor:
        return tf.convert_to_tensor(pickle.loads(value), dtype=TF_PRECISION) if value else None
