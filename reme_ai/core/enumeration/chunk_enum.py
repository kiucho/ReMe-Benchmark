"""Defines the types of data chunks used in streaming responses."""

from enum import Enum


class ChunkEnum(str, Enum):
    """Enumeration of possible chunk categories for stream processing."""

    # Internal reasoning or chain-of-thought process
    THINK = "think"

    # The final generated response content
    ANSWER = "answer"

    # Metadata or calls related to external tools
    TOOL = "tool"

    # Resource consumption and token usage statistics
    USAGE = "usage"

    # Error messages or exception details
    ERROR = "error"

    # Final signal indicating the completion of the stream
    DONE = "done"
