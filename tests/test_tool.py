"""Tests for tool operations including search and execution tools.

This module contains test functions for various tool operations such as
search tools (Dashscope, Mock, Tavily) and execution tools (Code, Shell).
"""

# pylint: disable=too-many-statements

import asyncio

from reme_ai.reme import ReMe

ReMe()


def test_search():
    """Test search tool operations.

    Tests DashscopeSearch, MockSearch, and TavilySearch operations
    with a sample query to verify they work correctly.
    """
    from reme_ai.tool.search import DashscopeSearch, MockSearch, TavilySearch

    query = "今天杭州的天气如何？"

    for op in [
        DashscopeSearch(),
        MockSearch(),
        TavilySearch(),
    ]:
        print("\n" + "=" * 60)
        print(f"Testing {op.__class__.__name__}")
        print("=" * 60)
        print(f"Query: {query}")
        asyncio.run(op.call(query=query))
        print(f"Output:\n{op.output}")


def test_execute():
    """Test code and shell execution tool operations.

    Tests ExecuteCode and ExecuteShell operations with various scenarios
    including successful execution, syntax errors, runtime errors, and
    invalid commands to verify error handling.
    """
    from reme_ai.tool.execute import ExecuteCode, ExecuteShell

    # Test ExecuteCode
    print("\n" + "=" * 60)
    print("Testing ExecuteCode")
    print("=" * 60)

    op = ExecuteCode()
    code_to_execute = "print('hello world')"
    print(f"Executing Python code: {code_to_execute}")
    asyncio.run(op.call(code=code_to_execute))
    print(f"Output:\n{op.output}")

    # Test ExecuteCode with more complex code
    print("\n" + "=" * 60)
    print("Testing ExecuteCode with calculation")
    print("=" * 60)

    op = ExecuteCode()
    code_to_execute = "result = sum(range(1, 11))\nprint(f'Sum of 1-10: {result}')"
    print(f"Executing Python code:\n{code_to_execute}")
    asyncio.run(op.call(code=code_to_execute))
    print(f"Output:\n{op.output}")

    # Test ExecuteShell
    print("\n" + "=" * 60)
    print("Testing ExecuteShell")
    print("=" * 60)

    op = ExecuteShell()
    command = "ls"
    print(f"Executing shell command: {command}")
    asyncio.run(op.call(command=command))
    print(f"Output:\n{op.output}")

    # Test ExecuteShell with echo
    print("\n" + "=" * 60)
    print("Testing ExecuteShell with echo")
    print("=" * 60)

    op = ExecuteShell()
    command = "echo 'Hello from shell!'"
    print(f"Executing shell command: {command}")
    asyncio.run(op.call(command=command))
    print(f"Output:\n{op.output}")

    # Test ExecuteCode with error (syntax error)
    print("\n" + "=" * 60)
    print("Testing ExecuteCode with syntax error (expected to fail)")
    print("=" * 60)

    op = ExecuteCode()
    code_to_execute = "print('missing closing quote)"
    print(f"Executing Python code with syntax error:\n{code_to_execute}")
    asyncio.run(op.call(code=code_to_execute))
    print(f"Output:\n{op.output}")

    # Test ExecuteCode with runtime error
    print("\n" + "=" * 60)
    print("Testing ExecuteCode with runtime error (expected to fail)")
    print("=" * 60)

    op = ExecuteCode()
    code_to_execute = "x = 1 / 0"
    print(f"Executing Python code with runtime error:\n{code_to_execute}")
    asyncio.run(op.call(code=code_to_execute))
    print(f"Output:\n{op.output}")

    # Test ExecuteCode with undefined variable
    print("\n" + "=" * 60)
    print("Testing ExecuteCode with undefined variable (expected to fail)")
    print("=" * 60)

    op = ExecuteCode()
    code_to_execute = "print(undefined_variable)"
    print(f"Executing Python code with undefined variable:\n{code_to_execute}")
    asyncio.run(op.call(code=code_to_execute))
    print(f"Output:\n{op.output}")

    # Test ExecuteShell with invalid command
    print("\n" + "=" * 60)
    print("Testing ExecuteShell with invalid command (expected to fail)")
    print("=" * 60)

    op = ExecuteShell()
    command = "this_command_does_not_exist"
    print(f"Executing invalid shell command: {command}")
    asyncio.run(op.call(command=command))
    print(f"Output:\n{op.output}")

    # Test ExecuteShell with command that returns non-zero exit code
    print("\n" + "=" * 60)
    print("Testing ExecuteShell with failing command (expected to fail)")
    print("=" * 60)

    op = ExecuteShell()
    command = "ls /nonexistent_directory_12345"
    print(f"Executing shell command that should fail: {command}")
    asyncio.run(op.call(command=command))
    print(f"Output:\n{op.output}")

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


def test_simple_chat():
    """Test simple chat operation.

    Tests the SimpleChat agent with a basic query to verify
    it can process and respond to user input.
    """
    from reme_ai.mem_agent.chat import SimpleChat

    op = SimpleChat()
    asyncio.run(op.call(query="你好"))
    print(op.output)


async def test_stream_chat():
    """Test streaming chat operation.

    Tests the StreamChat agent with a query to verify it can
    process and stream responses in real-time using async operations.
    """
    from reme_ai.mem_agent.chat import StreamChat
    from reme_ai.core.utils import execute_stream_task
    from reme_ai.core.context import RuntimeContext
    from asyncio import Queue

    op = StreamChat()
    context = RuntimeContext(query="你好，详细介绍一下自己", stream_queue=Queue())

    async def task():
        await op.call(context)
        await op.context.add_stream_done()

    async for chunk in execute_stream_task(
        stream_queue=context.stream_queue,
        task=asyncio.create_task(task()),
        task_name="test_stream_chat",
        as_bytes=False,
    ):
        print(chunk, end="")


if __name__ == "__main__":
    # test_search()
    # test_execute()
    # test_simple_chat()
    asyncio.run(test_stream_chat())
