from abc import ABC, abstractmethod
from typing import Iterable

from nodestream.subclass_registry import SubclassRegistry
from nodestream.pluggable import Pluggable

from .model import Content

CHUNKER_SUBCLASS_REGISTRY = SubclassRegistry()


@CHUNKER_SUBCLASS_REGISTRY.connect_baseclass
class Chunker(ABC, Pluggable):
    """Chunker is a mechanism to split a large document into smaller chunks.

    The chunker is used to split a large document into smaller chunks.
    The chunker is useful when the document is too large to be
    semantically meaningful as one piece of content.
    """

    entrypoint_name = "chunkers"

    @staticmethod
    def from_file_data(type, **chunker_kwargs) -> "Chunker":
        return CHUNKER_SUBCLASS_REGISTRY.get(type)(**chunker_kwargs)

    @abstractmethod
    def chunk(self, content: Content) -> Iterable[Content]: ...


class SplitOnDelimiterChunker(Chunker):
    def __init__(self, delimiter: str):
        self.delimiter = delimiter

    def chunk(self, content: Content) -> Iterable[Content]:
        for item in content.split_on_delimiter(self.delimiter):
            yield item
