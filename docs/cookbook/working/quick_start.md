---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Working Memory Demo

This demo showcases how to use ReMe's working memory capabilities with a ReAct agent. The working memory system automatically manages context by compressing and summarizing conversation history, enabling efficient long-context processing.

## Installation

### Install from PyPI (Recommended)

```bash
pip install reme-ai
```

### Install from Source

```bash
git clone https://github.com/agentscope-ai/ReMe.git
cd ReMe
pip install .
```

### Environment Configuration

Copy `example.env` to `.env` and modify the corresponding parameters:

```bash
FLOW_LLM_API_KEY=sk-xxxx
FLOW_LLM_BASE_URL=https://xxxx/v1
FLOW_EMBEDDING_API_KEY=sk-xxxx
FLOW_EMBEDDING_BASE_URL=https://xxxx/v1
```

## Starting the Services

Before running the demo, you need to start both the HTTP and MCP services:

### Start MCP Service

```bash
reme backend=mcp mcp.port=8002
```

The MCP service provides tools for working memory management including:
- `grep_working_memory`: Search for content in working memory
- `read_working_memory`: Read specific sections of working memory

### Start HTTP Service

```bash
reme backend=http http.port=8003
```

The HTTP service provides the flow execution endpoint for memory operations.

## Running the Demo

Once both services are running, execute the demo:


```bash
cd cookbook/working_memory
python work_memory_demo.py
```

### What the Demo Does

The demo simulates a scenario where:
1. A large README content is loaded (repeated 4 times to create a long context)
2. The agent needs to search through this content and extract specific information
3. Working memory automatically compresses the context from ~24,586 tokens to ~1,565 tokens (compression ratio: 0.06)
4. The agent can still accurately answer questions about the content

## Core Code Explanation

### ReactAgent with Working Memory (`react_agent_with_working_memory.py`)

#### 1. Agent Initialization

```python
class ReactAgent:
    def __init__(self, model_name="", max_steps: int = 50):
        # Use your own LLM class
        self.llm = OpenAICompatibleLLM(model_name=model_name)
        self.max_steps = max_steps
```

The agent is initialized with an LLM model and a maximum number of reasoning steps.

#### 2. Service Connection

```python
async with FastMcpClient("reme_mcp_server", {
    "type": "sse",
    "url": "http://0.0.0.0:8002/sse",
}) as mcp_client, HttpClient(base_url="http://localhost:8003") as http_client:
```

The agent connects to both:
- **MCP Client**: For tool execution (grep, read operations)
- **HTTP Client**: For flow execution (memory summarization)

#### 3. Tool Registration

```python
tool_calls = await mcp_client.list_tool_calls()

for tool_call in tool_calls:
    if tool_call.name in ["grep_working_memory", "read_working_memory"]:
        tool_dict[tool_call.name] = tool_call
```

The agent registers working memory tools that will be available to the LLM.

> Note: `summary_working_memory` is **not** an MCP tool.
> It is a **flow** exposed by the HTTP service and is invoked via `HttpClient.execute_flow`,
> as shown in the next section.

#### 4. Working Memory Summarization (Key Feature)

```python
result = await http_client.execute_flow("summary_working_memory",
                                        messages=[x.simple_dump() for x in messages],
                                        working_summary_mode="auto",
                                        compact_ratio_threshold=0.75,
                                        max_total_tokens=20000,
                                        max_tool_message_tokens=2000,
                                        group_token_threshold=None,
                                        keep_recent_count=1,
                                        store_dir="./test_working_memory")

messages = [Message(**x) for x in result.answer]
```

**This is the core of working memory management.** Before each LLM call:

- **`working_summary_mode="auto"`**: Automatically decides when to compress
- **`compact_ratio_threshold=0.75`**: Triggers compression when context exceeds 75% of max tokens
- **`max_total_tokens=20000`**: Maximum total tokens allowed
- **`max_tool_message_tokens=2000`**: Maximum tokens per tool message
- **`keep_recent_count=1`**: Keeps the most recent message uncompressed
- **`store_dir`**: Directory to store compressed memory

The summarization process:
1. Analyzes the current message history
2. Identifies compressible content (especially long tool outputs)
3. Compresses/summarizes old messages while preserving semantic information
4. Returns a condensed message list that maintains context

#### 5. ReAct Loop

```python
for i in range(self.max_steps):
    # Summarize working memory before each LLM call
    result = await http_client.execute_flow("summary_working_memory", ...)
    messages = [Message(**x) for x in result.answer]

    # LLM generates next action
    assistant_message = await self.llm.achat(messages=messages, tools=[...])
    messages.append(assistant_message)

    if not assistant_message.tool_calls:
        break

    # Execute tools
    for tool_call in assistant_message.tool_calls:
        result = await mcp_client.call_tool(tool_call.name,
                                           arguments=tool_call.argument_dict)
        messages.append(Message(role=Role.TOOL, content=result, ...))
```

The ReAct loop:
1. **Compress**: Summarize working memory to reduce context size
2. **Reason**: LLM decides what tool to use
3. **Act**: Execute the tool
4. **Observe**: Add tool result to messages
5. Repeat until task is complete or max steps reached

### Benefits of Working Memory

1. **Context Efficiency**: Reduces token usage by ~94% (24,586 → 1,565 tokens in the demo)
2. **Cost Reduction**: Lower token counts mean lower API costs
3. **Performance**: Faster inference with smaller contexts
4. **Scalability**: Handle much longer conversations and tool outputs
5. **Accuracy**: Maintains semantic information despite compression

## Model Configuration

The demo uses an OpenAI-compatible LLM configured via environment variables:

- **`FLOW_LLM_API_KEY` / `FLOW_LLM_BASE_URL`**: LLM API credentials and endpoint
- The model name is specified in `work_memory_demo.py`, for example:

```python
model_name = "qwen3-coder-30b-a3b-instruct"
agent = ReactAgent(model_name=model_name, max_steps=50)
```

You can change `model_name` to any model that your backend supports, as long as it follows the OpenAI-compatible API.

## Expected Output

When running the demo, you should see:
- Token count before compression: ~24,586 tokens
- Token count after compression: ~1,565 tokens
- Compression ratio: ~0.06 (6% of original size)
- The agent successfully answers the question about task memory performance in AppWorld

## Customization

You can customize the working memory behavior by adjusting parameters in the `summary_working_memory` call:

- **`compact_ratio_threshold`**: Lower values trigger compression earlier
- **`max_total_tokens`**: Adjust based on your model's context window
- **`max_tool_message_tokens`**: Control individual tool output size
- **`keep_recent_count`**: Keep more recent messages uncompressed for better context

## Troubleshooting

1. **Services not starting**: Ensure ports 8002 and 8003 are available
2. **Connection errors**: Verify both MCP and HTTP services are running
3. **API errors**: Check your `.env` file has valid API keys and endpoints
4. **Memory errors**: Adjust `max_total_tokens` based on your available memory
