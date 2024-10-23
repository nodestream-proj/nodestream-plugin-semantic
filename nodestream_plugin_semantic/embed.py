from abc import ABC, abstractmethod

from nodestream.subclass_registry import SubclassRegistry
from nodestream.pluggable import Pluggable

from .model import Content, Embedding


EMBEDDER_SUBCLASS_REGISTRY = SubclassRegistry()


@EMBEDDER_SUBCLASS_REGISTRY.connect_baseclass
class Embedder(ABC, Pluggable):
    """Embedder is a mechanism to embed content into a vector space."""

    entrypoint_name = "embedders"

    @classmethod
    def from_file_data(cls, type, **embedder_kwargs) -> "Embedder":
        cls.import_all()  # Import all embedders to register them.
        return EMBEDDER_SUBCLASS_REGISTRY.get(type)(**embedder_kwargs)

    @abstractmethod
    async def embed(self, content: Content) -> Embedding:
        """Embeds the content into a vector space.

        Args:
            content (Content): The content to embed.

        Returns:
            Embedding: The embedding of the content.
        """
        ...
