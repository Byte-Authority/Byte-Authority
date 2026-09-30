

import json
import os
import re
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from injecagent_runner import SpanEncoder, FAMILIES, parse_calls

PUBLIC_TO_INTERNAL = {
    "reserved": "A", "split": "B", "plaintext": "C", "matched": "E",
    "position_matched": "G", "preceding_matched": "H",
}

class TransportError(RuntimeError):
    pass

def _locate_spans(full: str, untrusted: list[str]) -> list[tuple[int, int]]:

    spans, cursor = [], 0
    for text in untrusted:
        if not text:
            continue
        for cand in (text, json.dumps(text, ensure_ascii=False), json.dumps(text)):
            i = full.find(cand, cursor)
            if i >= 0:
                spans.append((i, i + len(cand)))
                cursor = i + len(cand)
                break
        else:
            raise TransportError(
                f"untrusted span not found in the rendered prompt: {text[:80]!r}")
    return spans

def encode_by_provenance(enc: SpanEncoder, full: str, spans: list[tuple[int, int]],
                         span_encoder=None, trusted_untrusted=False) -> list[int]:

    for (a, b), (c, _d) in zip(spans, spans[1:]):
        if not a < b <= c:
            raise TransportError(f"untrusted spans overlap or are unordered: {spans}")

    ids, pos = [], 0
    for a, b in spans:
        ids += enc.enc_trusted(full[pos:a])
        seg = full[a:b]
        if span_encoder is not None:
            ids += span_encoder(seg)
        elif trusted_untrusted:
            ids += enc.enc_trusted(seg)
        else:
            ids += enc.enc_untrusted(seg)
        pos = b
    ids += enc.enc_trusted(full[pos:])

    if enc.decode(ids).encode() != full.encode():
        raise TransportError("decode(concat(ids)) != rendered prompt")
    return ids

_DEFAULT_TIMEOUT = int(os.environ.get("ADJ_REQUEST_TIMEOUT", "300"))

def complete_raw(base_url: str, model: str, ids: list[int], stop_ids,
                 max_tokens: int = 512, timeout: int = _DEFAULT_TIMEOUT) -> dict:

    payload = {"model": model, "prompt": ids, "temperature": 0,
               "max_tokens": max_tokens, "return_token_ids": True,
               "stop_token_ids": list(stop_ids)}
    req = urllib.request.Request(f"{base_url}/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.loads(r.read().decode())
    ch = (resp.get("choices") or [{}])[0]
    got = resp.get("prompt_token_ids") or ch.get("prompt_token_ids")

    if got is None:
        raise TransportError(
            "server did not return prompt_token_ids; cannot verify the model "
            "received the IDs we built. Check vLLM version / return_token_ids.")
    if list(got) != list(ids):
        n = sum(1 for x, y in zip(got, ids) if x != y)
        first = next((i for i, (x, y) in enumerate(zip(got, ids)) if x != y), None)
        raise TransportError(
            f"server prompt_token_ids differ from what we sent: "
            f"len {len(got)} vs {len(ids)}, {n} positional mismatches, "
            f"first at index {first}")

    return {"text": ch.get("text", ""),
            "finish_reason": ch.get("finish_reason"),
            "n_prompt_ids": len(ids),
            "error": None}

def arm_span_encoder(arm: str, enc: SpanEncoder, family: str):
    arm = PUBLIC_TO_INTERNAL.get(arm, arm)

    if arm == "A":
        return None, True
    if arm == "B":
        return None, False
    if arm == "C":

        return None, True
    if arm == "E":
        from quantity_matching import split_ordinary_to_match

        def enc_E(seg: str) -> list[int]:
            quota = len(enc.enc_untrusted(seg)) - len(enc.enc_trusted(seg))
            if quota <= 0:

                return enc.enc_trusted(seg)
            ids, _ = split_ordinary_to_match(enc, seg, quota, enc.protected)
            return ids

        return enc_E, True
    if arm in ("G", "H"):

        if arm == "G":
            from adjacent_matching import split_adjacent_to_match as split_fn
        else:
            from preceding_matching import split_preceding_to_match as split_fn

        def enc_pos(seg: str) -> list[int]:
            quota = len(enc.enc_untrusted(seg)) - len(enc.enc_trusted(seg))
            if quota <= 0:
                return enc.enc_trusted(seg)
            ids, _, _ = split_fn(enc, seg, quota, enc.protected)
            return ids

        return enc_pos, True
    raise ValueError(f"unknown arm {arm!r}")
