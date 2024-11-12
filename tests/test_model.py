from unittest.mock import Mock

from nodestream.model import DesiredIngestion, Node

from nodestream_plugin_semantic.model import Content, hash


def test_content_from_text():
    content_text = "test content"
    content = Content.from_text(content_text)
    assert content.content == content_text
    assert content.id == hash(content_text)
    assert content.parent is None


def test_content_add_metadata():
    content = Content.from_text("test content")
    content.add_metadata("key", "value")
    assert content.metadata == {"key": "value"}


def test_content_split_on_delimiter():
    content_text = "line1\nline2\nline3"
    content = Content.from_text(content_text)
    lines = list(content.split_on_delimiter("\n"))
    assert len(lines) == 3
    assert lines[0].content == "line1"
    assert lines[1].content == "line2"
    assert lines[2].content == "line3"
    assert all(line.parent == content for line in lines)


def test_content_assign_embedding():
    content = Content.from_text("test content")
    embedding = [0.1, 0.2, 0.3]
    content.assign_embedding(embedding)
    assert content.embedding == embedding


def test_content_apply_to_node():
    content = Content.from_text("test content")
    node = Mock(spec=Node)
    content.apply_to_node("test_type", node)
    node.type = "test_type"
    node.key_values.set_property.assert_called_with("id", content.id)
    node.properties.set_property.assert_any_call("content", content.content)


def test_content_make_ingestible():
    parent_content = Content.from_text("parent content")
    child_content = Content.from_text("child content", parent=parent_content)
    ingest = child_content.make_ingestible("test_type", "test_relationship")

    assert isinstance(ingest, DesiredIngestion)
    assert ingest.source.type == "test_type"
    ingest.source.key_values == {"id": child_content.id}
    ingest.source.properties == {"content": child_content.content}

    assert len(ingest.relationships) == 1
    relationship = ingest.relationships[0]
    assert relationship.relationship.type == "test_relationship"
    assert relationship.outbound == False
    assert relationship.to_node.type == "test_type"
    relationship.to_node.key_values == {"id": parent_content.id}
    relationship.to_node.properties == {"content": parent_content.content}
