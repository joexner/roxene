from __future__ import annotations

import dataclasses
import sqlalchemy
from sqlalchemy import CHAR, VARCHAR
from typing import List, Set

import roxene.tic_tac_toe as ttt


class Board(sqlalchemy.types.TypeDecorator):
    impl = CHAR(9)

    def process_bind_param(self, value: List[List[str]], dialect) -> str:
        if value is None:
            return None
        resultStr = ""
        for row in value:
            for sqVal in row:
                resultStr += (sqVal or ' ')

        return resultStr

    def process_result_value(self, value: str, dialect) -> List[List[str]]:
        if value is None:
            return None
        resultList: List[List[str]] = []
        for row in range(3):
            rowVal = []
            for col in range(3):
                sq_val = value[3 * row + col]
                rowVal.append(sq_val if sq_val in ('X', 'O') else None)
            resultList.append(rowVal)
        return resultList


@dataclasses.dataclass
class Point:
    row: int
    column: int


class OutcomeSet(sqlalchemy.types.TypeDecorator):
    impl = VARCHAR

    def process_bind_param(self, value: Set[ttt.Outcome], dialect) -> str:
        value_strings = map(lambda o: o.name, value)
        return ", ".join(value_strings)

    def process_result_value(self, value: str, dialect) -> Set[ttt.Outcome]:
        if value is None:
            return None
        resultSet: Set[ttt.Outcome] = set()
        for outcomeStr in value.split(", "):
            resultSet.add(ttt.Outcome[outcomeStr])
        return resultSet
