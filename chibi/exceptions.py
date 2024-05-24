class GptException(Exception):
    def __init__(
        self, provider: str, model: str = "unknown", detail: str = "Failed to receive response from the service"
    ) -> None:
        self.model = model
        self.provider = provider
        self.detail = detail

    def __str__(self) -> str:
        class_name = self.__class__.__name__
        return f"{class_name}(provider={self.provider!r}, model={self.model!r}, detail={self.detail!r})"

    def __repr__(self) -> str:
        return self.__str__()


class NoApiKeyProvidedError(GptException):
    ...


class NotAuthorizedError(GptException):
    ...


class ServiceRateLimitError(GptException):
    ...


class ServiceResponseError(GptException):
    ...


class ServiceConnectionError(GptException):
    ...
