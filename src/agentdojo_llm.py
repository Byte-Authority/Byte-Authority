

import json
import re
import time
from urllib.error import HTTPError
import sys
import threading
from collections.abc import Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionCall, FunctionsRuntime
from agentdojo.types import (ChatAssistantMessage, ChatMessage,
                             get_text_content_as_str, text_content_block_from_string)

from agentdojo_transport import (SpanEncoder, FAMILIES, TransportError,
                                 _locate_spans, encode_by_provenance, complete_raw,
                                 arm_span_encoder, parse_calls, PUBLIC_TO_INTERNAL)
from injecagent_runner import degeneracy_kind

_CALL_SHAPED = re.compile(
    r'<tool_call>|<\|python_tag\|>|\{\s*"name"\s*:|\{\s*"function"\s*:|'
    r'"parameters"\s*:|"arguments"\s*:|<seed:tool_call>|<function=', re.S)

import os
DUMP_TEXTS = os.environ.get("AGENTDOJO_DUMP_TEXTS") == "1"

AGENTDOJO_PARSER = {"llama": "llama_json"}

from xml_calls import _XML_CALL, parse_xml_calls

def _schema_types(prop: dict) -> set:

    ts = set()
    if "type" in prop:
        t = prop["type"]
        ts |= set(t) if isinstance(t, list) else {t}
    for alt in prop.get("anyOf", []) + prop.get("oneOf", []):
        ts |= _schema_types(alt)
    return ts

def coerce_xml_args(raw: dict, schema: dict) -> dict:

    props = (schema or {}).get("properties", {})
    out = {}
    for k, v in raw.items():
        ts = _schema_types(props.get(k, {}))
        if ts and ts <= {"string", "null"}:
            out[k] = v
            continue
        try:
            out[k] = json.loads(v)
        except (ValueError, TypeError):
            out[k] = v
    return out

MAX_CALLS_PER_TURN = {"llama": 1}

_ENC_CACHE: dict[str, SpanEncoder] = {}
_ENC_LOCK = threading.Lock()

def get_encoder(path: str) -> SpanEncoder:
    with _ENC_LOCK:
        if path not in _ENC_CACHE:
            _ENC_CACHE[path] = SpanEncoder(path)
        return _ENC_CACHE[path]

_POS_CACHE: dict = {}
_POS_LOCK = threading.Lock()

def _span_at_marker(enc, arm, span_enc, seg):

    key = (id(enc), arm, seg)
    with _POS_LOCK:
        if key in _POS_CACHE:
            return _POS_CACHE[key]
    from span_position import at_marker
    sa = enc.enc_trusted(seg)
    if len(enc.enc_untrusted(seg)) <= len(sa):
        ok = True
    else:
        ok = at_marker(enc, sa, span_enc(seg), enc.protected,
                       "after" if arm == "G" else "before")
    with _POS_LOCK:
        if len(_POS_CACHE) > 20000:
            _POS_CACHE.clear()
        _POS_CACHE[key] = ok
    return ok

def _tools_for_template(runtime: FunctionsRuntime) -> list[dict]:

    out = []
    for f in runtime.functions.values():
        out.append({"type": "function", "function": {
            "name": f.name,
            "description": (f.description or "")[:400],
            "parameters": f.parameters.model_json_schema(),
        }})
    return out

ARGS_AS_MAPPING = {"glm", "seed"}

def _to_template_messages(messages: Sequence[ChatMessage],
                          family: str = "qwen") -> tuple[list[dict], list[str]]:

    msgs, untrusted = [], []
    for m in messages:
        role = m["role"]
        if role == "tool":
            if m.get("error"):
                content = json.dumps({"error": m["error"]})
            else:
                result = get_text_content_as_str(m["content"]) if m.get("content") else ""
                content = json.dumps({"result": result if result != "None" else "Success"})
            untrusted.append(content)
            msgs.append({"role": "tool", "content": content})
        elif role == "assistant":
            calls = []
            for tc in (m.get("tool_calls") or []):
                args = (tc.args if family in ARGS_AS_MAPPING
                        else json.dumps(tc.args))
                calls.append({"id": tc.id or "call_0", "type": "function",
                              "function": {"name": tc.function,
                                           "arguments": args}})
            entry = {"role": "assistant",
                     "content": get_text_content_as_str(m["content"]) if m.get("content") else ""}
            if calls:
                entry["tool_calls"] = calls
            msgs.append(entry)
        else:
            msgs.append({"role": role,
                         "content": get_text_content_as_str(m["content"]) if m.get("content") else ""})
    return msgs, untrusted

class RawIDLLM(BasePipelineElement):
    def __init__(self, base_url: str, model: str, tokenizer_path: str,
                 family: str = "qwen", arm: str = "A", max_tokens: int = 512,
                 trace: list | None = None) -> None:
        self.base_url = base_url
        self.model = model
        self.family = family
        self.condition = arm
        self.arm = PUBLIC_TO_INTERNAL.get(arm, arm)
        self.max_tokens = max_tokens
        self.enc = get_encoder(tokenizer_path)
        self.span_enc, self.trusted_untrusted = arm_span_encoder(arm, self.enc, family)

        self.trace = trace if trace is not None else []

    def query(self, query: str, runtime: FunctionsRuntime, env: Env = EmptyEnv(),
              messages: Sequence[ChatMessage] = [], extra_args: dict = {}):
        tmpl_msgs, untrusted = _to_template_messages(messages, self.family)
        tools = _tools_for_template(runtime)

        full = self.enc.tk.apply_chat_template(
            tmpl_msgs, tools=tools, tokenize=False, add_generation_prompt=True)
        spans = _locate_spans(full, untrusted)
        ids = encode_by_provenance(self.enc, full, spans,
                                   span_encoder=self.span_enc,
                                   trusted_untrusted=self.trusted_untrusted)

        count_ok = protected_ok = None
        if self.arm in ("E", "G", "H"):
            ids_b = encode_by_provenance(self.enc, full, spans, span_encoder=None,
                                         trusted_untrusted=False)
            ids_a = encode_by_provenance(self.enc, full, spans, span_encoder=None,
                                         trusted_untrusted=True)
            prot = self.enc.protected
            count_ok = len(ids) == len(ids_b)
            protected_ok = (sum(t in prot for t in ids) == sum(t in prot for t in ids_a))

        position_ok = None
        if self.arm in ("G", "H"):
            position_ok = all(_span_at_marker(self.enc, self.arm, self.span_enc, full[a:b])
                              for a, b in spans)

        r, err, n_retries, verified, exhausted = None, None, 0, False, False
        for attempt in range(3):
            try:
                r = complete_raw(self.base_url, self.model, ids,
                                 FAMILIES[self.family]["stop_ids"],
                                 max_tokens=self.max_tokens)
                err, verified = None, True
                break
            except TransportError:

                raise
            except HTTPError as e:

                body = ""
                try:
                    body = e.read().decode()[:400]
                except Exception:
                    pass
                if e.code == 400 and "maximum context length" in body:
                    exhausted = True
                    err = None
                    break
                err = f"HTTPError: HTTP Error {e.code}: {e.reason} :: {body[:160]}"
                n_retries = attempt + 1
                if attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
            except Exception as e:
                err = f"{type(e).__name__}: {str(e)[:200]}"
                n_retries = attempt + 1
                if attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
        if r is None:
            r = {"text": "", "finish_reason": "context_exhausted" if exhausted
                 else None, "n_prompt_ids": len(ids)}

        degen_kind = degeneracy_kind(r.get("text") or "")
        self.trace.append({
            "condition": self.condition,
            "turn": len([m for m in messages if m["role"] == "assistant"]),
            "n_prompt_ids": len(ids), "n_untrusted_spans": len(spans),
            "count_matches_B": count_ok, "protected_kept": protected_ok,
            "position_ok": position_ok,

            "prompt_ids_verified": verified,
            "n_retries": n_retries,

            "context_exhausted": exhausted,

            "degeneracy_kind": degen_kind,
            "degenerate": degen_kind is not None,
            "finish_reason": r.get("finish_reason"), "error": err,
            "text_head": (r.get("text") or "")[:400],

            "text_full": (r.get("text") or "") if DUMP_TEXTS else None,
            "n_calls": None,
        })

        if exhausted:

            self.trace[-1]["n_calls"] = 0
            self.trace[-1]["n_calls_dropped"] = 0
            out = ChatAssistantMessage(
                role="assistant",
                content=[text_content_block_from_string("")], tool_calls=[])
            return query, runtime, env, [*messages, out], extra_args

        text = r.get("text", "") or ""

        calls = []
        if self.family in _XML_CALL:
            parsed = []
            for c in parse_xml_calls(text, self.family):
                fn = runtime.functions.get(c["name"])
                schema = fn.parameters.model_json_schema() if fn is not None else {}
                parsed.append({"name": c["name"],
                               "arguments": coerce_xml_args(c["raw_args"], schema)})
        else:
            parsed = parse_calls(text, AGENTDOJO_PARSER.get(self.family,
                                                            FAMILIES[self.family]["parser"]))
        for c in parsed:
            if not c.get("name"):
                continue
            a = c.get("arguments", c.get("args")) or {}
            if isinstance(a, str):
                try:
                    a = json.loads(a)
                except Exception:
                    a = {}
            if not isinstance(a, dict):
                a = {}
            calls.append(FunctionCall(function=c["name"], args=a))

        n_args_missing = 0
        for fc in calls:
            fn = runtime.functions.get(fc.function)
            if fn is None:
                continue
            req = fn.parameters.model_json_schema().get("required", [])
            if req and not any(k in fc.args for k in req):
                n_args_missing += 1
        self.trace[-1]["n_args_missing"] = n_args_missing

        n_parsed = len(calls)
        looks_like_call = bool(_CALL_SHAPED.search(text))
        self.trace[-1]["looks_like_call"] = looks_like_call
        self.trace[-1]["parse_miss"] = looks_like_call and n_parsed == 0
        if looks_like_call and n_parsed == 0:
            self.trace[-1]["parse_miss_text"] = text[:600]
        cap = MAX_CALLS_PER_TURN.get(self.family)
        if cap is not None and n_parsed > cap:
            calls = calls[:cap]
        self.trace[-1]["n_calls"] = len(calls)
        self.trace[-1]["n_calls_dropped"] = n_parsed - len(calls)
        out = ChatAssistantMessage(role="assistant",
                                   content=[text_content_block_from_string(text.strip())],
                                   tool_calls=calls)
        return query, runtime, env, [*messages, out], extra_args
