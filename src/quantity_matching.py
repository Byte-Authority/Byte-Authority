import os

import sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from injecagent_runner import SpanEncoder, FAMILIES

def split_ordinary_to_match(enc, text, target_extra, protected):

    base = enc.enc_trusted(text)

    pieces, cur = [], []
    for t in base:
        if t in protected:
            if cur: pieces.append(("ord", cur)); cur = []
            pieces.append(("ctl", [t]))
        else:
            cur.append(t)
    if cur: pieces.append(("ord", cur))

    out, extra = [], 0
    for kind, toks in pieces:
        if kind == "ctl" or extra >= target_extra:
            out.extend(toks); continue
        txt = enc.decode(toks)
        canon = len(toks)
        need = target_extra - extra

        best = None
        for k in range(0, len(txt) + 1):
            buf = []
            for ch in txt[:k]:
                buf.extend(enc.enc_trusted(ch))
            if k < len(txt):
                buf.extend(enc.enc_trusted(txt[k:]))
            if enc.decode(buf) != txt:
                continue
            add = len(buf) - canon
            if best is None or abs(add - need) < abs(best[0] - need):
                best = (add, buf)
            if add >= need:
                break
        if best is None:
            out.extend(toks); continue
        add, buf = best
        out.extend(buf); extra += max(0, add)
    return out, extra

def main():
    enc = SpanEncoder(FAMILIES["qwen"]["model_path"])
    prot = enc.protected
    from injecagent_runner import arm_payloads
    cases = json.load(open("data/third_party/injecagent/data/test_cases_dh_base.json"))
    import random
    rng = random.Random(0); idx = list(range(len(cases))); rng.shuffle(idx)

    ok = 0; rows = []
    for ci in idx[:5]:
        p = arm_payloads(cases[ci]["Attacker Instruction"], "qwen", seed=0)
        a = enc.enc_trusted(p["A"]["text"])
        b = enc.enc_untrusted(p["B"]["text"])
        need = len(b) - len(a)
        e, got = split_ordinary_to_match(enc, p["A"]["text"], need, prot)
        byte_ok = enc.decode(e) == enc.decode(a)
        ctl_atomic = len(prot & set(e)) == len(prot & set(a))
        rows.append({"case": ci, "nA": len(a), "nB": len(b), "nE": len(e),
                     "need_extra": need, "byte_identical": byte_ok,
                     "control_tokens_still_atomic": ctl_atomic})
        print(f"  case {ci}: A={len(a)} B={len(b)} E={len(e)}  "
              f"target_extra={need}  byte_ok={byte_ok}  ctl_atomic={ctl_atomic}")
        ok += byte_ok and ctl_atomic
    print(f"\nvalid arm-E constructions: {ok}/5")
    json.dump(rows, open("data/results/arm_e_construction_check.json","w"), indent=2)

if __name__ == "__main__":
    main()

def split_both_to_match(enc, text, target_total, protected):

    base = enc.enc_trusted(text)
    spans, pos = [], 0
    for t in base:
        w = len(enc.decode([t]))
        if t in protected:
            spans.append((pos, pos + w))
        pos += w

    ordinary, cur = [], 0
    for a, b in spans + [(len(text), len(text))]:
        ordinary.extend(range(cur, a))
        cur = b

    def build(split_at):
        out, i = [], 0
        buf = []
        for j, ch in enumerate(text):
            if j in split_at:
                if buf:
                    out.extend(enc.enc_untrusted("".join(buf)))
                    buf = []
                out.extend(enc.enc_untrusted(ch))
            else:
                buf.append(ch)
        if buf:
            out.extend(enc.enc_untrusted("".join(buf)))
        return out

    chosen = set()
    ids = build(chosen)
    if len(ids) == target_total:
        return ids, True
    for j in ordinary:
        if len(ids) >= target_total:
            break
        trial = build(chosen | {j})
        if trial is None or enc.decode(trial) != text:
            continue
        if len(trial) > target_total:
            continue
        chosen.add(j)
        ids = trial
        if len(ids) == target_total:
            return ids, True
    return ids, len(ids) == target_total
