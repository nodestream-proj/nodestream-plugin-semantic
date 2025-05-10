from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional

from nodestream.model import DesiredIngestion
from nodestream.pipeline import Extractor, Transformer
from nodestream.pipeline.value_providers import (
    JmespathValueProvider,
    ProviderContext,
    ValueProvider,
)
from nodestream.schema import (
    Cardinality,
    ExpandsSchema,
    GraphObjectSchema,
    SchemaExpansionCoordinator,
)

from .chunk import Chunker
from .content_types import PLAIN_TEXT_ALIAS, ContentType
from .embed import Embedder
from .model import Content

from .model import DeclarativeJsonSchema
from .inference import InferenceRequestor
from logging import getLogger
from math import ceil
import re


DEFAULT_ID = JmespathValueProvider.from_string_expression("id")
DEFAULT_CONTENT = JmespathValueProvider.from_string_expression("content")
DEFAULT_NODE_TYPE = "Content"
DEFAULT_CHILD_RELATIONSHIP_TYPE = "HAS_CHILD"


class ChunkContent(Transformer):
    """Transforms a document into smaller chunks."""

    @classmethod
    def from_file_data(cls, **chunker_kwargs) -> "ChunkContent":
        return cls(Chunker.from_file_data(**chunker_kwargs))

    def __init__(self, chunker: Chunker):
        self.chunker = chunker

    async def transform_record(self, record: Content):
        for chunk in self.chunker.chunk(record):
            yield chunk


class EmbedContent(Transformer):
    """Transforms a document into an embedded document."""

    @classmethod
    def from_file_data(cls, **embedder_kwargs) -> "EmbedContent":
        return cls(Embedder.from_file_data(**embedder_kwargs))

    def __init__(self, embedder: Embedder):
        self.embedder = embedder

    async def transform_record(self, content: Content) -> Content:
        emebedding = await self.embedder.embed(content)
        content.assign_embedding(emebedding)
        return content


class DocumentExtractor(Extractor):
    """Extracts documents from files.

    The DocumentExtractor reads files from the given paths and
    extracts the content of the files. The content is then
    returned as a Content object.
    """

    @classmethod
    def from_file_data(
        cls,
        globs: List[str],
        content_types: Optional[List[str]] = None,
    ):
        paths = [Path(file) for glob_ in globs for file in glob(glob_)]
        content_types = [
            ContentType.by_name(content_type)
            for content_type in content_types or [PLAIN_TEXT_ALIAS]
        ]
        return cls(paths, content_types)

    def __init__(self, paths: List[Path], content_types: List[ContentType]):
        self.paths = paths
        self.content_types = content_types

    def content_type(self, file: Path) -> ContentType:
        for content_type in self.content_types:
            if content_type.is_supported(file):
                return content_type

        raise ValueError(f"Unsupported file: {file}")

    def read(self, file: Path) -> str:
        return self.content_type(file).read(file)

    async def extract_records(self):
        for file in self.paths:
            yield Content.from_text(self.read(file))


class ConvertToContent(Transformer):
    """Converts a record into a Content object.

    Records are expected to be dictionaries with the following keys:
    - content: The text content of the document.
    - id: The unique identifier of the document.

    You can override the expected keys by supplying the `content_provider`
    and `id_provider` arguments.

    The transformer converts the record into a Content object.
    """

    def __init__(
        self,
        content: Optional[ValueProvider] = None,
        id: Optional[ValueProvider] = None,
        metadata: Optional[Dict[str, ValueProvider]] = None,
    ):
        self.content_provider = content or DEFAULT_CONTENT
        self.id_provider = id or DEFAULT_ID
        self.metadata_providers = metadata or {}

    async def transform_record(self, record: dict) -> Content:
        context = ProviderContext.fresh(record)
        content = Content(
            id=self.id_provider.single_value(context),
            content=self.content_provider.single_value(context),
        )
        for key, provider in self.metadata_providers.items():
            content.add_metadata(key, provider.single_value(context))
        return content


class ContentInterpreter(Transformer, ExpandsSchema):
    """Interprets the content of a document.

    The ContentInterpreter interprets the content of a document.
    The content is expected to be a Content object.

    The transformer interprets the content and yields the interpreted content.
    """

    def __init__(
        self,
        node_type: str = DEFAULT_NODE_TYPE,
        child_relationship_type: str = DEFAULT_CHILD_RELATIONSHIP_TYPE,
    ):
        self.node_type = node_type
        self.child_relationship_type = child_relationship_type

    async def transform_record(self, content: Content) -> DesiredIngestion:
        return content.make_ingestible(
            node_type=self.node_type,
            relationship_type=self.child_relationship_type,
        )

    def expand_node_type(self, node_type: GraphObjectSchema):
        node_type.add_key("id")

    def expand_relationship_type(self, _: GraphObjectSchema):
        pass

    def expand_schema(self, coordinator: SchemaExpansionCoordinator):
        coordinator.on_node_schema(self.expand_node_type, self.node_type)
        coordinator.on_relationship_schema(
            self.expand_relationship_type, self.child_relationship_type
        )
        coordinator.connect(
            self.node_type,
            self.node_type,
            self.child_relationship_type,
            Cardinality.MANY,
            Cardinality.SINGLE,
        )


CHARS_PER_TOKEN = 2
DEFAULT_PROMPT_LOCATION = "nodestream_plugin_semantic/prompts/text2json.txt"
PROMPT_FILE_ERROR = (
    "The prompt must contain formatting fields for the {text} and the {schema}."
)


class TextToJson(Transformer):
    _required_parameters = set(["schema", "text"])

    def __init__(
        self,
        schema: dict,
        inference_requestor_kwargs: dict,
        discard_invalid: bool = False,
        prompt_location: str = DEFAULT_PROMPT_LOCATION,
    ):
        self.schema = DeclarativeJsonSchema.from_file_data(schema)
        self.inference_requestor = InferenceRequestor.from_file_data(
            **inference_requestor_kwargs
        )
        self.discard_invalid = discard_invalid
        self.prompt = self.read_prompt_from_file(prompt_location)
        self.logger = getLogger(name=self.__class__.__name__)

    def read_prompt_from_file(self, path: str) -> str:
        with open(path, "r") as prompt_file:
            prompt_string = prompt_file.read()
            variables = re.findall(r"{(\w+)}", prompt_string)
            if not set(variables) == self._required_parameters:
                raise ValueError(PROMPT_FILE_ERROR)
            return prompt_file

    async def transform_record(self, data: dict) -> Any:
        text = data.pop("content")
        additional_args = data

        """
            Handle the chunking and partitioning of the text/text stream to fit the 
            context window. Make an assumption that the length of text*2 will be under 
            the maximum token limit of the model. TODO find ways to officially 
            determine token length.
        """
        string_size = len(text)
        chunk_size = self.inference_requestor.context_window * CHARS_PER_TOKEN
        chunk_count = int(ceil(string_size / (chunk_size)))
        for index in range(chunk_count):
            begin = index * chunk_size
            end = min(string_size, (index + 1) * chunk_size)
            substring = text[begin:end]

            # Create the prompt using the schema and the truncated text.
            # Handle entity resolution orchestration here.
            prompt = self.prompt.format(
                schema=self.schema.prompt_representation, text=substring
            )

            # Execute the prompt using the inference requestor.
            result: list[dict] | dict = await self.inference_requestor.execute_prompt(
                prompt
            )

            # Parse the response and yield the records.
            # The response should be a JSON object.
            if isinstance(result, list):
                for piece in result:
                    if self.schema.validate(piece):
                        piece.update(additional_args)
                        yield piece
                    elif not self.discard_invalid:
                        self.logger.info(f"Invalid item passed: {piece}.")
                        piece.update(additional_args)
                        yield piece

            elif isinstance(result, dict):
                if self.schema.validate(result):
                    result.update(additional_args)
                    yield result
                elif not self.discard_invalid:
                    self.logger.info(f"Invalid item passed: {result}.")
                    result.update(additional_args)
                    yield result
            else:
                raise ValueError(f"Invalid result format: {result}.")
