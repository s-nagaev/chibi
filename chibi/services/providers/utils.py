import inspect
from typing import Any, Callable, Coroutine, Type, TypeVar

T = TypeVar("T")
M = TypeVar("M", bound=Callable[..., Coroutine[Any, Any, Any]])


def decorate_async_methods(decorator: Callable[[M], M]) -> Callable[[Type[T]], Type[T]]:
    def decorate(cls: Type[T]) -> Type[T]:
        for attr in cls.__dict__:
            if inspect.iscoroutinefunction(getattr(cls, attr)):
                original_func = getattr(cls, attr)
                decorated_func = decorator(original_func)
                setattr(cls, attr, decorated_func)
        return cls

    return decorate
