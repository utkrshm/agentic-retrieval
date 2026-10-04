"""Tree-sitter based code chunking."""

from coderet.chunking.treesitter import (
    EXTENSIONS,
    MAX_CHARS,
    Unit,
    chunk_file,
    chunk_source,
    detect_language,
    embed_text,
    iter_source_files,
)

__all__ = [
    "EXTENSIONS",
    "MAX_CHARS",
    "Unit",
    "chunk_file",
    "chunk_source",
    "detect_language",
    "embed_text",
    "iter_source_files",
]
