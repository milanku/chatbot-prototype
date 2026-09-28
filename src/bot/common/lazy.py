from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class Lazy(Generic[T]):
    def __init__(self, factory: Callable[[], T]) -> None:
        self._factory = factory
        self._instance: T | None = None

    def get(self) -> T:
        if self._instance is None:
            self._instance = self._factory()
        return self._instance
