from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from nodestream.model import DesiredIngestion

from nodestream_plugin_semantic.model import Content
from nodestream_plugin_semantic.pipeline import (
    ChunkContent,
    ContentInterpreter,
    ConvertToContent,
    DocumentExtractor,
    EmbedContent,
)


@pytest.mark.asyncio
async def test_chunk_content():
    chunker = MagicMock()
    chunker.chunk.return_value = [
        Content(id="1", content="chunk1"),
        Content(id="2", content="chunk2"),
    ]
    transformer = ChunkContent(chunker)
    record = Content(id="0", content="original content")
    chunks = [chunk async for chunk in transformer.transform_record(record)]
    assert len(chunks) == 2
    assert chunks[0].content == "chunk1"
    assert chunks[1].content == "chunk2"


@pytest.mark.asyncio
async def test_embed_content():
    embedder = AsyncMock()
    embedder.embed.return_value = "embedded content"
    transformer = EmbedContent(embedder)
    content = Content(id="0", content="original content")
    result = await transformer.transform_record(content)
    assert result.content == "original content"
    assert result.embedding == "embedded content"


def test_document_extractor():
    paths = [Path("file1.txt"), Path("file2.txt")]
    content_type = MagicMock()
    content_type.is_supported.return_value = True
    content_type.read.return_value = "file content"
    with patch(
        "nodestream_plugin_semantic.pipeline.glob",
        return_value=["file1.txt", "file2.txt"],
    ), patch(
        "nodestream_plugin_semantic.pipeline.ContentType.by_name",
        return_value=content_type,
    ):
        extractor = DocumentExtractor.from_file_data(globs=["*.txt"])
        assert len(extractor.paths) == 2
        assert extractor.read(paths[0]) == "file content"


@pytest.mark.asyncio
async def test_document_extractor_extract_records():
    content_type = MagicMock()
    content_type.is_supported.return_value = True
    content_type.read.return_value = "file content"
    with patch(
        "nodestream_plugin_semantic.pipeline.glob", return_value=["file1.txt"]
    ), patch(
        "nodestream_plugin_semantic.pipeline.ContentType.by_name",
        return_value=content_type,
    ):
        extractor = DocumentExtractor.from_file_data(globs=["*.txt"])
        records = [record async for record in extractor.extract_records()]
        assert len(records) == 1
        assert records[0].content == "file content"


@pytest.mark.asyncio
async def test_convert_to_content():
    record = {"id": "1", "content": "some content"}
    transformer = ConvertToContent()
    content = await transformer.transform_record(record)
    assert content.id == "1"
    assert content.content == "some content"


@pytest.mark.asyncio
async def test_content_interpreter():
    content = Content(id="1", content="some content")
    transformer = ContentInterpreter()
    desired_ingestion = await transformer.transform_record(content)
    assert isinstance(desired_ingestion, DesiredIngestion)


def test_content_interpreter_expand_schema():
    transformer = ContentInterpreter()
    coordinator = MagicMock()
    transformer.expand_schema(coordinator)
    coordinator.on_node_schema.assert_called()
    coordinator.on_relationship_schema.assert_called()
    coordinator.connect.assert_called()
