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

# Message Reload Ops

## GrepOp

### Purpose

Provides a text search capability for locating specific content within offloaded files by pattern matching. This operation enables case-insensitive search within a single file, making it easy to find specific lines in tool messages or compressed groups.

### Functionality

- Searches for literal text patterns (case-insensitive) within a single file
- Limits result count to avoid overwhelming output (default: 50 matches)
- Returns matching lines with file path, line number, and content
- Ideal for locating specific content within known offloaded files

### Parameters

- `file_path` (string, **required**):
  - The path to the file to search in
  - Can be an absolute or relative path
  - Must be a valid file path (not a directory)
  - Examples:
    - `/workspace/context_store/tool_call_123.txt`
    - `./context_store/compressed_group_0.json`

- `pattern` (string, **required**):
  - The text pattern to search for in the file
  - Search is case-insensitive
  - Searched as a literal string (special regex characters are escaped)
  - Examples: `"stored in"`, `"error message"`, `"function_name"`

- `limit` (number, optional, default: `50`):
  - Maximum number of matching lines to return
  - Stops searching after reaching the limit
  - Useful for large files to avoid token overflow
  - Example: `100` returns at most 100 matching lines

### Return Value

The operation returns search results with matching lines:
- Each match is formatted as: `file_path:line_number:line_content`
- Returns up to `limit` matches
- If no matches found, returns a message indicating no matches
- Each match shows the complete line containing the pattern

Example: Searching for `"error"` in `/workspace/context_store/tool_call_123.txt` with limit 50 returns matching lines like:
```
/workspace/context_store/tool_call_123.txt:45:Error: Connection timeout
/workspace/context_store/tool_call_123.txt:78:Warning: Retrying after error
```

## ReadFileOp

### Purpose

Reads and returns the content of offloaded files, enabling on-demand access to compacted tool messages and compressed conversation history. Supports efficient pagination for handling large files.

### Functionality

- Reads file content from specified path (absolute or relative)
- Supports pagination with offset and limit for reading specific line ranges
- Uses efficient `sed` command for line-based reading
- Works with text files
- Essential for retrieving full content of compacted tool messages
- Enables access to original message groups before compression

### Parameters

- `file_path` (string, **required**):
  - The path to the file to read
  - Can be absolute or relative path
  - Path will be expanded and resolved automatically
  - Examples:
    - `/workspace/context_store/tool_call_123.txt`
    - `./context_store/compressed_group_0.json`
    - `~/context_store/message.txt`

- `offset` (number, **required** but has default):
  - The 0-based line number to start reading from
  - If not provided or 0, starts from the beginning of the file
  - Used in combination with `limit` for pagination
  - Example: `0` starts from the first line, `100` starts from line 100

- `limit` (number, **required** but has default):
  - Maximum number of lines to read from the offset
  - If not provided, defaults to 1,000,000 (reads to end of file)
  - Used with `offset` to implement pagination
  - Example: `100` reads up to 100 lines from the offset

### Return Value

The operation returns the file content as a string:
- Content of the specified line range (from `offset` to `offset + limit`)
- Lines are returned without trailing newlines
- Empty string if the specified range is beyond the file's content
- Error message if file not found or cannot be read

Example: Reading `/workspace/context_store/tool_call_123.txt` with `offset=0` and `limit=100` returns the first 100 lines of the file.

## Usage Pattern: Combining Grep and ReadFile

For a complete working example of how to use these operations in practice, please refer to:
[test_agentic_retrieve_op.py](../../test_op/test_agentic_retrieve_op.py)

This test file demonstrates:
- How to configure the system prompt to guide AI in using Grep and ReadFile operations
- Real-world usage scenarios with message offload and reload
- Proper parameter settings for `AgenticRetrieveOp` with working memory
- Best practices for combining these operations in a retrieval workflow

