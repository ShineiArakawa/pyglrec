from __future__ import annotations

import enum

# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# Other utility classes


class StrEnum(str, enum.Enum):
    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return self.value

    def __list__(self) -> list[str]:
        return [e.value for e in self.__class__]

    @classmethod
    def _items(cls) -> list[str]:
        return [e.value for e in cls]


class EasyDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

# ----------------------------------------------------------------------------------------------------------------------------------------------------------
