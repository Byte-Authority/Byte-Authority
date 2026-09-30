

import difflib

def changed_regions(a, x):
    sm = difflib.SequenceMatcher(a=a, b=x, autojunk=False)
    return [(i1, i2) for tag, i1, i2, _j1, _j2 in sm.get_opcodes() if tag != "equal"]

def _changed_mask(a, regions):
    m = [False] * len(a)
    for s, e in regions:
        for k in range(s, e):
            m[k] = True
    return m

def _n_untouched_splittable(enc, a, lo, hi, mask):
    return sum(1 for k in range(lo, hi) if not mask[k] and len(enc.decode([a[k]])) > 1)

def gap_after_marker(enc, a, start, prot, mask=None):
    mask = mask or [False] * len(a)
    j = start - 1
    while j >= 0 and a[j] not in prot:
        j -= 1
    return None if j < 0 else _n_untouched_splittable(enc, a, j + 1, start, mask)

def gap_before_marker(enc, a, end, prot, mask=None):
    mask = mask or [False] * len(a)
    j = end
    while j < len(a) and a[j] not in prot:
        j += 1
    return None if j >= len(a) else _n_untouched_splittable(enc, a, end, j, mask)

def region_gaps(enc, a, x, prot, side):

    reg = changed_regions(a, x)
    mask = _changed_mask(a, reg)
    out = []
    for s, e in reg:
        out.append(gap_after_marker(enc, a, s, prot, mask) if side == "after"
                   else gap_before_marker(enc, a, e, prot, mask))
    return out

def at_marker(enc, a, x, prot, side):

    g = region_gaps(enc, a, x, prot, side)
    return bool(g) and all(d == 0 for d in g)
