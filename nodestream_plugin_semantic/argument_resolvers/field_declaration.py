from nodestream.pipeline.argument_resolvers import ArgumentResolver
from typing import Any
from nodestream.file_io import LazyLoadedTagSafeLoader, LazyLoadedArgument
from enum import Enum


TYPE_DECLARATION_ERROR = (
    "Type conversion was unsuccesful. Attempted to convert {data} to type: {type}"
)


class TypeDeclaration(str, Enum):
    STRING = "string"
    FLOAT = "float"
    INTEGER = "integer"
    BOOLEAN = "boolean"

    _PYTHON_TRANSLATOR = {
        "string": str,
        "float": float,
        "integer": int,
        "boolean": bool,
    }

    def convert(self, data: Any) -> str | int | bool | list:
        try:
            return self._PYTHON_TRANSLATOR[self.value](data)
        except Exception:
            raise ValueError(TYPE_DECLARATION_ERROR.format(data=data, type=self.value))


def wrap_declared_tag(self, node):
    value = self.construct_mapping(node)
    return LazyLoadedArgument(node.tag[1:], value)


LazyLoadedTagSafeLoader.add_constructor("!declare", wrap_declared_tag)


class FieldDeclaration(ArgumentResolver, alias="declare"):
    @staticmethod
    def resolve_argument(value: dict):
        return FieldDeclaration(
            type=value.get("type", None),
            description=value.get("description", None),
            examples=value.get("examples", []),
            required=value.get("required", False),
        )

    def __init__(
        self,
        type: TypeDeclaration | None = None,
        description: str | None = None,
        examples: list[str] = [],
        required: bool = False,
    ) -> None:
        self.type = type
        self.description = description
        self.examples = examples
        self.required = required

    @staticmethod
    def field(**kwargs) -> str:
        return [f"{key}={value}" for key, value in kwargs.items()]

    def __str__(self) -> str:
        return "; ".join(
            self.field(
                type=self.type,
                description=self.description,
                examples=[{",".join(self.examples)}],
                required=self.required,
            )
        )

    def __repr__(self) -> str:
        return self.__str__()

    def validate(self, data: Any) -> bool:
        if self.required and data is None:
            return False
        if not self.type:
            return True
        return self.type.convert(data)
