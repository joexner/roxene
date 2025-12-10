import logging

from sqlalchemy import create_engine, Engine

from roxene import EntityBase

_engine: Engine = None


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        logging.info("Creating engine")
        _engine = create_engine("sqlite://",)
        # _engine = create_engine("sqlite:///roxene.db",)
        # _engine = create_engine("postgresql://postgres:postgres@localhost:5432/test")
    logging.info("Initializing database")
    EntityBase.metadata.drop_all(_engine)
    EntityBase.metadata.create_all(_engine)
    logging.info("Done initializing database")
    return _engine
