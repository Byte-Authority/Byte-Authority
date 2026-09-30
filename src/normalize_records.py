import json
import sys

LABELS = {
    "A": "reserved", "B": "split", "C": "plaintext", "D": "perturbed",
    "E": "matched", "G": "position_matched", "H": "preceding_matched",
    "P": "character_split", "Q": "character_matched", "W": "embedding_near",
    "X": "embedding_far", "Y": "embedding_near_matched", "M": "lookalike_matched",
    "N": "lookalike", "N1": "fixed_variant_1", "N2": "fixed_variant_2",
    "N3": "fixed_variant_3", "N4": "fixed_variant_4", "N5": "fixed_variant_5",
    "N6": "fixed_variant_6",
}

def public_label(value):
    if value in LABELS: return LABELS[value]
    if value.startswith("L") and value[1:].isdigit(): return "searched_variant_" + value[1:]
    return value

def convert(src, dst):
    with open(src) as inp, open(dst, "w") as out:
        for line in inp:
            row=json.loads(line)
            if "arm" in row:
                row["condition"]=public_label(row.pop("arm"))
            elif "condition" in row:
                row["condition"]=public_label(row["condition"])
            out.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

if __name__ == "__main__":
    if len(sys.argv) != 3: raise SystemExit("usage: normalize_records.py INPUT OUTPUT")
    convert(sys.argv[1], sys.argv[2])
