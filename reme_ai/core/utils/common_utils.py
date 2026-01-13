"""Common utility functions"""

import asyncio
from collections.abc import AsyncGenerator, Coroutine
from typing import Any

from loguru import logger

from ..enumeration import ChunkEnum
from ..schema import StreamChunk


def run_coro_safely(coro: Coroutine[Any, Any, Any]) -> Any | asyncio.Task[Any]:
    """Run a coroutine in the current event loop or a new one if none exists."""
    try:
        # Attempt to retrieve the event loop associated with the current thread
        loop = asyncio.get_running_loop()

    except RuntimeError:
        # Start a new event loop to run the coroutine to completion
        return asyncio.run(coro)

    else:
        # Schedule the coroutine as a background task in the active loop
        return loop.create_task(coro)


async def execute_stream_task(
    stream_queue: asyncio.Queue,
    task: asyncio.Task,
    task_name: str | None = None,
    as_bytes: bool = False,
) -> AsyncGenerator[str | bytes, None]:
    """
    Core stream flow execution logic.

    Handles streaming from a queue while monitoring the task completion.
    Properly manages errors and resource cleanup.

    Args:
        stream_queue: Queue to receive StreamChunk objects from
        task: Background task executing the flow
        task_name: Optional flow name for logging purposes
        as_bytes: If True, yield bytes for HTTP responses; if False, yield strings

    Yields:
        SSE-formatted data chunks (either str or bytes based on as_bytes)
    """
    done_msg = b"data:[DONE]\n\n" if as_bytes else "data:[DONE]\n\n"

    try:
        while True:
            # Wait for next chunk or check if task failed
            get_chunk = asyncio.create_task(stream_queue.get())
            done, _ = await asyncio.wait({get_chunk, task}, return_when=asyncio.FIRST_COMPLETED)

            if get_chunk in done:
                chunk: StreamChunk = get_chunk.result()
                if chunk.done:
                    yield done_msg
                    break

                data = f"data:{chunk.model_dump_json()}\n\n"
                yield data.encode() if as_bytes else data
            else:
                # Task finished unexpectedly or raised exception
                await task
                yield done_msg
                break

    except Exception as e:
        log_msg = f"Stream error in {task_name}: {e}" if task_name else f"Stream error: {e}"
        logger.exception(log_msg)

        err = StreamChunk(chunk_type=ChunkEnum.ERROR, chunk=str(e), done=True)
        err_data = f"data:{err.model_dump_json()}\n\n"
        yield err_data.encode() if as_bytes else err_data
        yield done_msg

    finally:
        # Ensure task is cancelled if still running to avoid resource leaks
        if not task.done():
            task.cancel()
