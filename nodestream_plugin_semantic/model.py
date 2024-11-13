import hashlib
from dataclasses import dataclass
from typing import Iterable, List, Optional

from nodestream.model import DesiredIngestion, Node, Relationship

Embedding = List[float | int]
CONTENT_NODE_TYPE_ID_PROPERTY = "id"


def hash(content: str) -> str:
    sha1 = hashlib.sha1()
    sha1.update(content.encode())
    return sha1.hexdigest()


@dataclass(slots=True)
class Content:
    """Content is a piece of text.

    Content is a piece of text that can be embedded into a vector space.
    """

    id: str
    content: str
    parent: Optional["Content"] = None
    embedding: Optional[Embedding] = None
    metadata: Optional[dict] = None

    @classmethod
    def from_text(
        cls,
        content: str,
        parent: Optional["Content"] = None,
    ) -> "Content":
        return cls(id=hash(content), content=content, parent=parent)

    def add_metadata(self, key: str, value: str):
        if not self.metadata:
            self.metadata = {}
        self.metadata[key] = value

    def split_on_delimiter(self, delimiter: str) -> Iterable["Content"]:
        for line in self.content.split(delimiter):
            yield Content.from_text(line, parent=self)

    def assign_embedding(self, embedding: Embedding):
        self.embedding = embedding

    def apply_to_node(self, node_type: str, node: Node):
        node.type = node_type
        node.key_values.set_property(CONTENT_NODE_TYPE_ID_PROPERTY, self.id)
        node.properties.set_property("content", self.content)
        if self.embedding:
            node.properties.set_property("embedding", self.embedding)
        if self.metadata:
            for key, value in self.metadata.items():
                node.properties.set_property(key, value)

    def make_ingestible(
        self, node_type: str, relationship_type: str
    ) -> DesiredIngestion:
        ingest = DesiredIngestion()
        self.apply_to_node(node_type, ingest.source)

        if self.parent:
            self.parent.apply_to_node(node_type, related := Node())
            relationship = Relationship(type=relationship_type)
            ingest.add_relationship(
                related_node=related, relationship=relationship, outbound=False
            )

        return ingest
