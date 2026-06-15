class GptException(Exception):
    def __init__(
        self,
        provider: str = "unknown",
        model: str = "unknown",
        detail: str = "Failed to receive response from the service",
    ) -> None:
        self.model = model
        self.provider = provider
        self.detail = detail

    def __str__(self) -> str:
        class_name = self.__class__.__name__
        return f"{class_name}(provider={self.provider!r}, model={self.model!r}, detail={self.detail!r})"

    def __repr__(self) -> str:
        return self.__str__()


class NoApiKeyProvidedError(GptException): ...


class NotAuthorizedError(GptException): ...


class ServiceRateLimitError(GptException): ...


class ServiceResponseError(GptException): ...


class NoResponseError(GptException): ...


class ServiceConnectionError(GptException): ...


class NoModelSelectedError(GptException): ...


class NoProviderSelectedError(GptException): ...


class NoAccountIDSetError(GptException): ...


class RecursionLimitExceeded(GptException):
    def __init__(
        self,
        exceeded_limit: int,
        provider: str = "unknown",
        model: str = "unknown",
        detail: str = "Failed to receive response from the service",
    ) -> None:
        self.model = model
        self.provider = provider
        self.detail = detail
        self.exceeded_limit = exceeded_limit


class MemoryException(Exception):
    """Base exception for long-term memory operations."""

    def __init__(self, detail: str = "Memory operation failed") -> None:
        self.detail = detail
        super().__init__(detail)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(detail={self.detail!r})"

    def __repr__(self) -> str:
        return self.__str__()


class ChromaConnectionError(MemoryException):
    """Failed to connect to ChromaDB."""

    ...


class ChromaCollectionError(MemoryException):
    """Error accessing or creating a ChromaDB collection."""

    ...


class ChromaArchiveError(MemoryException):
    """Failed to archive messages to ChromaDB."""

    ...


class ChromaSearchError(MemoryException):
    """Failed to perform semantic search in ChromaDB."""

    ...
