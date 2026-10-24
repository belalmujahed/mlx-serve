"""Streaming text splitter and tool-call parsing for local model output."""

import json
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

TOOL_START = "<tool_call>"
THINK_START = "<think>"
THINK_END = "</think>"

_TOOL_BLOCK_RE = re.compile(r"<tool_call>(.*?)(?:</tool_call>|$)", re.S)
# Qwen3-Coder style: <function=name><parameter=key>value</parameter></function>
_FUNCTION_RE = re.compile(r"<function=([^>\s]+)>(.*?)(?:</function>|$)", re.S)
_PARAM_RE = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?(?:</parameter>|$)", re.S)
# GLM style: name<arg_key>k</arg_key><arg_value>v</arg_value>
_GLM_ARG_RE = re.compile(
    r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.S
)
_NAME_RE = re.compile(r"^[\w.\-]+$")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


class StreamParser:
    """Splits generated text into thinking, plain text, stop-sequence hits and
    tool calls. Feed returns a list of ("thinking" | "text", str) pieces.

    Text that could be the beginning of a marker is held back until it can be
    classified.
    """

    def __init__(
        self, stop_sequences: Optional[List[str]] = None, in_thinking: bool = False
    ):
        self.stops = [s for s in (stop_sequences or []) if s]
        self.pending = ""
        self.tool_buf = ""
        self.in_tool = False
        self.in_thinking = in_thinking
        self.stop_hit: Optional[str] = None

    def _markers(self) -> List[str]:
        if self.in_thinking:
            return [THINK_END]
        return self.stops + [TOOL_START, THINK_START]

    def feed(self, text: str) -> List[Tuple[str, str]]:
        if self.in_tool:
            self.tool_buf += text
            return []

        self.pending += text
        out: List[Tuple[str, str]] = []

        while self.pending:
            kind = "thinking" if self.in_thinking else "text"
            markers = self._markers()

            best = None
            for marker in markers:
                idx = self.pending.find(marker)
                if idx >= 0 and (best is None or idx < best[0]):
                    best = (idx, marker)

            if best is not None:
                idx, marker = best
                if idx:
                    out.append((kind, self.pending[:idx]))
                rest = self.pending[idx + len(marker) :]
                if marker == THINK_END:
                    self.in_thinking = False
                    self.pending = rest.lstrip("\n")
                elif marker == THINK_START:
                    self.in_thinking = True
                    self.pending = rest.lstrip("\n")
                elif marker == TOOL_START:
                    self.in_tool = True
                    self.tool_buf = marker + rest
                    self.pending = ""
                    return out
                else:
                    self.stop_hit = marker
                    self.pending = ""
                    return out
                continue

            hold = 0
            for marker in markers:
                for k in range(min(len(marker) - 1, len(self.pending)), hold, -1):
                    if self.pending.endswith(marker[:k]):
                        hold = k
                        break
            cut = len(self.pending) - hold
            if cut:
                out.append((kind, self.pending[:cut]))
            self.pending = self.pending[cut:]
            break

        return out

    def finish(self) -> List[Tuple[str, str]]:
        """Flush held-back text once generation has ended."""
        if self.in_tool or self.stop_hit or not self.pending:
            return []
        kind = "thinking" if self.in_thinking else "text"
        out, self.pending = [(kind, self.pending)], ""
        return out


def _coerce(value: str, prop_schema: Optional[Dict[str, Any]]) -> Any:
    if prop_schema and prop_schema.get("type") == "string":
        return value
    try:
        return json.loads(value)
    except ValueError, TypeError:
        return value


def _schema_props(tools: Dict[str, Dict[str, Any]], name: str) -> Dict[str, Any]:
    return (tools.get(name) or {}).get("properties", {}) or {}


def _parse_one(body: str, tools: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    body = body.strip()
    if not body:
        return None

    if body.startswith("{"):
        try:
            data = json.loads(body)
        except ValueError:
            return None
        name = data.get("name")
        args = data.get("arguments", data.get("input", data.get("parameters", {})))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                args = {}
        if name and isinstance(args, dict):
            return {"name": name, "input": args}
        return None

    match = _FUNCTION_RE.search(body)
    if match:
        name, inner = match.group(1), match.group(2)
        props = _schema_props(tools, name)
        args = {
            key: _coerce(val, props.get(key)) for key, val in _PARAM_RE.findall(inner)
        }
        return {"name": name, "input": args}

    name = body.split("<arg_key>", 1)[0].strip().split("\n", 1)[0].strip()
    if _NAME_RE.match(name):
        props = _schema_props(tools, name)
        args = {
            key.strip(): _coerce(val.strip(), props.get(key.strip()))
            for key, val in _GLM_ARG_RE.findall(body)
        }
        if args or name in tools:
            return {"name": name, "input": args}
    return None


def parse_tool_calls(
    buffer: str, tools: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Parse every <tool_call> block in buffer into {"name", "input"} dicts.

    `tools` maps tool name to its JSON input schema.
    """
    calls = []
    for body in _TOOL_BLOCK_RE.findall(buffer):
        call = _parse_one(body, tools)
        if call:
            calls.append(call)
    return calls
