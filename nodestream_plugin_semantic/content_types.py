from abc import ABC, abstractmethod
from typing import Iterable
from pathlib import Path

from nodestream.subclass_registry import SubclassRegistry


CONTENT_TYPE_SUBCLASS_REGISTRY = SubclassRegistry()
PLAIN_TEXT_ALIAS = "plain_text"
PLAIN_TEXT_EXTENSIONS = {".txt", ".md"}


@CONTENT_TYPE_SUBCLASS_REGISTRY.connect_baseclass
class ContentType(ABC):
    """Describes the mechanism to read a file of a specific content type."""

    @classmethod
    def all(cls) -> Iterable["ContentType"]:
        cls.import_all()  # Import all embedders to register them.
        for sub in CONTENT_TYPE_SUBCLASS_REGISTRY.all_subclasses():
            yield sub()

    @classmethod
    def by_name(cls, name: str) -> "ContentType":
        cls.import_all()  # Import all embedders to register them.
        return CONTENT_TYPE_SUBCLASS_REGISTRY.get(name)()

    @abstractmethod
    def is_supported(self, file_path: Path) -> bool:
        """Returns True if the file extension is supported.

        Args:
            file_path (Path): The file path to check.

        Returns:
            bool: True if the file extension is supported, False otherwise.
        """
        ...

    @abstractmethod
    def read(self, file_path: Path) -> str:
        """Reads the content of the file.

        Args:
            file_path (Path): The file path to read.

        Returns:
            str: The content of the file.
        """
        ...


class PlainText(ContentType, alias=PLAIN_TEXT_ALIAS):
    """Reads plain text files."""

    def is_supported(self, file_path: Path) -> bool:
        return file_path.suffix in PLAIN_TEXT_EXTENSIONS

    def read(self, file_path: Path) -> str:
        with file_path.open("r") as f:
            return f.read()
