class ToolException(Exception): ...


class NoUserInterfaceProvidedException(ToolException):
    def __init__(self, message: str | None = None) -> None:
        self.message = message or "This function requires user interface object to be automatically provided"
        super().__init__(self.message)


class LoopDetectedException(Exception): ...
