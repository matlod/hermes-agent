"""Tests for orithra_ground_truth_metrics (ORITHRA PATCH, not upstream).

Guards the ground-truth context metrics fed to the ``pre_api_request`` hook and
consumed by orithra's nats-bridge plugin / Context Lens.  These numbers are
computed from the real pre-sanitisation ``api_messages`` and ``agent.tools``
precisely because the hook's request payload is clamped by
``_sanitize_hook_payload`` at 50k chars.

The behaviour here is FROZEN — it is a byte-for-byte extraction of the original
inline block in ``run_conversation``.  The quirks asserted below (falsy content
counts as 0; tool_calls measured as JSON) are deliberate, not bugs: changing
them would silently shift values already stored downstream.
"""

import hashlib
import json

from agent.orithra_gt_metrics import orithra_ground_truth_metrics

TOOLS = [
    {"type": "function", "function": {"name": "bash", "description": "run a command"}},
    {"type": "function", "function": {"name": "read", "description": "read a file"}},
]


def test_returns_exactly_the_four_hook_kwarg_names():
    """The dict is splatted into _invoke_hook, so the key names are the contract."""
    assert set(orithra_ground_truth_metrics([], None)) == {
        "gt_role_chars",
        "gt_role_counts",
        "gt_tool_schemas_chars",
        "gt_tool_schema_hash",
    }


def test_empty_messages_and_no_tools():
    out = orithra_ground_truth_metrics([], None)
    assert out["gt_role_chars"] == {}
    assert out["gt_role_counts"] == {}
    assert out["gt_tool_schemas_chars"] == 0
    assert out["gt_tool_schema_hash"] == ""


def test_empty_tool_list_is_treated_as_no_tools():
    out = orithra_ground_truth_metrics([{"role": "user", "content": "hi"}], [])
    assert out["gt_tool_schemas_chars"] == 0
    assert out["gt_tool_schema_hash"] == ""
    assert out["gt_role_chars"] == {"user": 2}


def test_multi_role_messages_accumulate_per_role():
    messages = [
        {"role": "system", "content": "abc"},
        {"role": "user", "content": "de"},
        {"role": "user", "content": "f"},
        {"role": "assistant", "content": "ghij"},
        {"content": "no role key"},  # falls back to "unknown"
    ]
    out = orithra_ground_truth_metrics(messages, None)
    assert out["gt_role_chars"] == {
        "system": 3,
        "user": 3,
        "assistant": 4,
        "unknown": 11,
    }
    assert out["gt_role_counts"] == {
        "system": 1,
        "user": 2,
        "assistant": 1,
        "unknown": 1,
    }


def test_tool_calls_are_counted_as_json_into_the_same_role_bucket():
    tool_call = {"id": "call_1", "function": {"name": "bash", "arguments": '{"cmd":"ls"}'}}
    messages = [{"role": "assistant", "content": "hi", "tool_calls": [tool_call]}]
    out = orithra_ground_truth_metrics(messages, None)
    expected = len("hi") + len(json.dumps(tool_call, default=str))
    assert out["gt_role_chars"] == {"assistant": expected}
    assert out["gt_role_counts"] == {"assistant": 1}


def test_tool_calls_counted_even_when_content_is_none():
    tool_call = {"id": "c", "function": {"name": "x", "arguments": "{}"}}
    messages = [{"role": "assistant", "content": None, "tool_calls": [tool_call]}]
    out = orithra_ground_truth_metrics(messages, None)
    assert out["gt_role_chars"] == {"assistant": len(json.dumps(tool_call, default=str))}


def test_falsy_content_counts_as_zero_chars_frozen_quirk():
    """``len(str(c)) if c else 0`` — 0 and [] contribute nothing. Deliberate."""
    messages = [
        {"role": "user", "content": 0},
        {"role": "user", "content": ""},
        {"role": "user", "content": []},
    ]
    out = orithra_ground_truth_metrics(messages, None)
    assert out["gt_role_chars"] == {"user": 0}
    assert out["gt_role_counts"] == {"user": 3}


def test_non_string_content_uses_str_length():
    content = [{"type": "text", "text": "describe"}]
    out = orithra_ground_truth_metrics([{"role": "user", "content": content}], None)
    assert out["gt_role_chars"] == {"user": len(str(content))}


def test_tool_schema_hash_is_canonical_sorted_sha256():
    out = orithra_ground_truth_metrics([], TOOLS)
    expected = "sha256:" + hashlib.sha256(
        json.dumps(TOOLS, sort_keys=True, default=str).encode()
    ).hexdigest()
    assert out["gt_tool_schema_hash"] == expected
    assert out["gt_tool_schemas_chars"] == sum(
        len(json.dumps(t, default=str)) for t in TOOLS
    )


def test_tool_schema_hash_is_stable_under_key_reordering():
    reordered = [
        {"function": {"description": "run a command", "name": "bash"}, "type": "function"},
        {"function": {"description": "read a file", "name": "read"}, "type": "function"},
    ]
    assert (
        orithra_ground_truth_metrics([], reordered)["gt_tool_schema_hash"]
        == orithra_ground_truth_metrics([], TOOLS)["gt_tool_schema_hash"]
    )


def test_tool_schema_hash_changes_when_a_tool_changes():
    changed = [dict(TOOLS[0]), {"type": "function", "function": {"name": "write"}}]
    assert (
        orithra_ground_truth_metrics([], changed)["gt_tool_schema_hash"]
        != orithra_ground_truth_metrics([], TOOLS)["gt_tool_schema_hash"]
    )


def test_non_serialisable_objects_fall_back_to_str():
    class Weird:
        def __repr__(self):
            return "<Weird>"

    out = orithra_ground_truth_metrics(
        [{"role": "assistant", "content": Weird()}], [{"o": Weird()}]
    )
    assert out["gt_role_chars"] == {"assistant": len("<Weird>")}
    assert out["gt_tool_schemas_chars"] > 0


def test_large_context_is_measured_in_full_not_clamped():
    """The whole point of the fork patch: no 50k sanitiser ceiling here."""
    messages = [
        {"role": "system", "content": "S" * 30000},
        {"role": "user", "content": "U" * 90000},
    ]
    out = orithra_ground_truth_metrics(messages, None)
    assert out["gt_role_chars"] == {"system": 30000, "user": 90000}
