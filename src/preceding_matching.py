import os

import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def split_preceding_to_match(enc, text, target_extra, protected):

    base = enc.enc_trusted(text)
    pieces, cur = [], []
    for t in base:
        if t in protected:
            if cur:
                pieces.append(["ord", cur, False])
                cur = []
            pieces.append(["ctl", [t], False])
        else:
            cur.append(t)
    if cur:
        pieces.append(["ord", cur, False])
    for i in range(len(pieces) - 1):
        if pieces[i][0] == "ord" and pieces[i + 1][0] == "ctl":
            pieces[i][2] = True

    order = [i for i, p in enumerate(pieces) if p[0] == "ord" and p[2]]
    order += [i for i, p in enumerate(pieces) if p[0] == "ord" and not p[2]]

    extra = 0
    ended_at_control = None
    for i in order:
        if extra >= target_extra:
            break
        _, toks, before_ctl = pieces[i]
        txt = enc.decode(toks)
        canon = len(toks)
        need = target_extra - extra
        best = None
        for k in range(0, len(txt) + 1):
            head, tail = txt[:len(txt) - k], txt[len(txt) - k:]
            buf = list(enc.enc_trusted(head)) if head else []
            for ch in tail:
                buf.extend(enc.enc_trusted(ch))
            if enc.decode(buf) != txt:
                continue
            add = len(buf) - canon
            if best is None or abs(add - need) < abs(best[0] - need):
                best = (add, buf)
            if add >= need:
                break
        if best is None or best[0] <= 0:
            continue
        add, buf = best
        pieces[i][1] = buf
        extra += add
        if ended_at_control is None:
            ended_at_control = before_ctl

    out = []
    for _, toks, _ in pieces:
        out.extend(toks)
    return out, extra, bool(ended_at_control)

def arm_h_ids(enc, payload_text, n_target_extra, family):

    ids, _, _ = split_preceding_to_match(enc, payload_text, n_target_extra,
                                         enc.protected)
    return ids
