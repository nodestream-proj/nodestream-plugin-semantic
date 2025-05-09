from abc import ABC, abstractmethod, abstractproperty

from nodestream.pluggable import Pluggable
from nodestream.subclass_registry import SubclassRegistry

INFERENCE_SUBCLASS_REGISTRY = SubclassRegistry()


@INFERENCE_SUBCLASS_REGISTRY.connect_baseclass
class InferenceRequestor(ABC, Pluggable):
    """Embedder is a mechanism to embed content into a vector space."""

    entrypoint_name = "inferencers"

    @classmethod
    def from_file_data(cls, type, **inference_kwargs) -> "InferenceRequestor":
        cls.import_all()  # Import all inferencers to register them.
        return INFERENCE_SUBCLASS_REGISTRY.get(type)(**inference_kwargs)


    @abstractproperty
    def context_window(self) -> int:
        """
            The context window of the model.
        """
        ...

    @abstractmethod
    async def execute_prompt(self, prompt: str) -> str:
        """
            Executes the given prompt and returns the response from an arbitrary model.
        """
        ...
