from .file_storage import FileStorage
from .local_storage import LocalFileStorage
from .telegram_storage import TelegramFileStorage

__all__ = [
    "FileStorage",
    "LocalFileStorage",
    "TelegramFileStorage",
]

from chibi.config import application_settings
from chibi.services.interface import TelegramInterface, UserInterface


def get_file_storage(interface: UserInterface) -> FileStorage:
    if application_settings.selected_file_storage == "telegram" and isinstance(interface, TelegramInterface):
        return TelegramFileStorage(interface=interface)
    else:
        raise ValueError(
            f"Unsupported file storage: {application_settings.selected_file_storage} or user interface type."
        )
