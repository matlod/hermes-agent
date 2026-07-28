"""ORITHRA PATCH — ground-truth context metrics for the ``pre_api_request`` hook.

This module is **not upstream**.  It exists so that orithra's single divergence
from ``NousResearch/hermes-agent`` lives in a file upstream has never heard of,
leaving only a one-line call site inside ``agent/conversation_loop.py``.  A new
file cannot conflict during a rebase; a 25-line block wedged into the middle of
``run_conversation`` conflicts on every upstream refresh.

Why this cannot be a plugin
---------------------------
``pre_api_request`` receives the request body via
``HermesAgent._api_request_payload_for_hook`` -> ``_sanitize_hook_payload``
(``run_agent.py``).  That sanitiser caps the serialised payload at
``HERMES_PLUGIN_PAYLOAD_MAX_CHARS`` (default 50,000) and, above the cap,
re-serialises with strings clipped to 1,000 chars and sequences clipped to 50
elements — or, failing that, replaces the whole body with a ``_truncated``
preview.  Large contexts are precisely what these metrics measure, so a
plugin-side recomputation would be silently wrong exactly when it matters most.
The numbers must be taken from the real ``api_messages`` / ``agent.tools``
objects, in-process, before sanitisation.  Hence a fork patch.

What breaks if this is dropped
------------------------------
**The failure is silent.**  The consumer signature at
``hermes/plugins/nats-bridge/__init__.py`` declares these as
``gt_role_chars: Optional[Dict[str, int]] = None``,
``gt_role_counts: Optional[Dict[str, int]] = None``,
``gt_tool_schemas_chars: int = 0``, ``gt_tool_schema_hash: str = ""``, and it
absorbs unknown kwargs via ``**_kwargs``.  If this patch is lost in a rebase,
nothing raises, no test fails, and no log line complains — the hook simply
fires with the defaults, and Context Lens ground truth (read at
``services/media-gateway/gateway.py``, keys ``ground_truth.role_chars`` /
``ground_truth.tool_schemas_chars`` / ``ground_truth.tool_schema_hash``)
quietly degrades to zeros and empty strings.  Verify after every upstream
merge that ``grep -rn "ORITHRA PATCH" agent/conversation_loop.py`` still
matches.

Behaviour is frozen
-------------------
This is a byte-for-byte extraction of the original inline block (commit
``ac13dd2d5``).  The quirks below are deliberate and must not be "cleaned up",
because downstream stored values would shift and historical rows would stop
comparing:

* Content chars are ``len(str(content)) if content else 0`` — falsy content
  (``None``, ``""``, ``0``, ``[]``) contributes 0, even though ``str(0)`` has
  length 1.
* ``tool_calls`` entries are measured as ``len(json.dumps(tc, default=str))``,
  i.e. the JSON encoding including punctuation, and are added to the same
  role bucket as the message's content.
* The tool-schema hash is exactly
  ``"sha256:" + sha256(json.dumps(tools, sort_keys=True, default=str))``.
* No defensive ``or []`` around ``api_messages``: if it were ever ``None`` the
  original raised, and the caller's ``except Exception: pass`` suppressed the
  entire hook invocation.  That behaviour is preserved on purpose.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, Optional


def orithra_ground_truth_metrics(
    api_messages: Iterable[Dict[str, Any]],
    tools: Optional[Iterable[Any]],
) -> Dict[str, Any]:
    """Compute ground-truth context metrics from the real pre-sanitisation objects.

    Args:
        api_messages: the exact message list about to be sent to the provider.
        tools: ``agent.tools`` (may be ``None`` or empty).

    Returns:
        A dict whose keys are exactly the ``pre_api_request`` kwarg names, so the
        call site can splat it: ``**orithra_ground_truth_metrics(...)``.
    """
    gt_role_chars: Dict[str, int] = {}
    gt_role_counts: Dict[str, int] = {}
    for _am in api_messages:
        _role = _am.get("role", "unknown")
        _content = _am.get("content", "")
        _char_count = len(str(_content)) if _content else 0
        for _tc in (_am.get("tool_calls") or []):
            _char_count += len(json.dumps(_tc, default=str))
        gt_role_chars[_role] = gt_role_chars.get(_role, 0) + _char_count
        gt_role_counts[_role] = gt_role_counts.get(_role, 0) + 1

    gt_tool_schemas_chars = (
        sum(len(json.dumps(t, default=str)) for t in tools)
        if tools else 0
    )

    if tools:
        gt_tool_schema_hash = "sha256:" + hashlib.sha256(
            json.dumps(tools, sort_keys=True, default=str).encode()
        ).hexdigest()
    else:
        gt_tool_schema_hash = ""

    return {
        "gt_role_chars": gt_role_chars,
        "gt_role_counts": gt_role_counts,
        "gt_tool_schemas_chars": gt_tool_schemas_chars,
        "gt_tool_schema_hash": gt_tool_schema_hash,
    }
