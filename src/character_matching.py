

def split_per_character(enc, text, protected):

    base = enc.enc_trusted(text)

    out, n_markers, n_marker_tokens = [], 0, 0
    buf = []
    for t in base:
        if t in protected:
            if buf:
                out.extend(enc.enc_untrusted(enc.decode(buf)))
                buf = []
            surface = enc.decode([t])

            pieces = []
            for ch in surface:
                pieces.extend(enc.enc_untrusted(ch))
            if enc.decode(pieces) != surface:
                raise ValueError(f"per-character re-encoding changed the bytes of "
                                 f"{surface!r}")
            out.extend(pieces)
            n_markers += 1
            n_marker_tokens += len(pieces)
        else:
            buf.append(t)
    if buf:
        out.extend(enc.enc_untrusted(enc.decode(buf)))

    if enc.decode(out) != text:
        raise ValueError("arm P is not byte-identical to arm A")
    return out, n_markers, n_marker_tokens

def arm_p_ids(enc, payload_text, family=None):

    ids, _, _ = split_per_character(enc, payload_text, enc.protected)
    return ids
