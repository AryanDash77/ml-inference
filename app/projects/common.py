"""
Helpers shared by every project module.

build_schema() turns a project's FIELDS table into a pydantic request model
with one named field per column, so each project only has to describe its
columns once and gets validation + the /docs form for free.
"""

from typing import Annotated, Literal

from pydantic import Field, StringConstraints, create_model


def build_schema(name: str, fields: list[dict]):
    """FIELDS table -> pydantic model.

    Each row: name, kind ("int" | "float" | "choice" | "text"), desc, and
    optionally default / required / min / max / choices / example.
    A "text" column is free text: use min_length / max_length (not min / max).
    Leading and trailing spaces are stripped, so blank text is rejected.
    """
    pydantic_fields = {}
    for f in fields:
        rules = {}
        if f["kind"] == "choice":
            py_type = Literal[tuple(f["choices"])]  # type: ignore[valid-type]
        elif f["kind"] == "text":
            py_type = Annotated[
                str,
                StringConstraints(
                    strip_whitespace=True,
                    min_length=f.get("min_length", 1),
                    max_length=f.get("max_length"),
                ),
            ]
        else:
            py_type = int if f["kind"] == "int" else float
            if "min" in f:
                rules["ge"] = f["min"]
            if "max" in f:
                rules["le"] = f["max"]

        if "example" in f:
            rules["examples"] = [f["example"]]

        # "..." means the caller must send this column; otherwise use the default.
        default = ... if f.get("required") else f["default"]
        pydantic_fields[f["name"]] = (
            py_type,
            Field(default, description=f["desc"], **rules),
        )
    return create_model(name, **pydantic_fields)

