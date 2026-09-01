from darwin_neg_router.tool_parser import parse_tool_calls, split_thinking


def test_splits_thinking_and_tool_call() -> None:
    raw = """<think>inspect first</think>
I will check it.
<tool_call>
<function=read_file>
<parameter=path>
src/main.py
</parameter>
<parameter=lines>
[1, 20]
</parameter>
</function>
</tool_call>"""
    reasoning, visible = split_thinking(raw)
    content, calls = parse_tool_calls(visible)
    assert reasoning == "inspect first"
    assert content == "I will check it."
    assert calls[0]["function"]["name"] == "read_file"
    assert calls[0]["function"]["arguments"] == '{"path": "src/main.py", "lines": [1, 20]}'

