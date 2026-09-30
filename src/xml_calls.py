

import re

_XML_CALL = {
    "glm": re.compile(r"<tool_call>\s*([A-Za-z_][A-Za-z0-9_.\-]*)\s*(.*?)(?:</tool_call>|$)", re.S),
    "seed": re.compile(r"<function=([A-Za-z_][A-Za-z0-9_.\-]*)\s*>(.*?)(?:</function>|$)", re.S),
}
_XML_ARG = {
    "glm": re.compile(r"<arg_key>\s*(.*?)\s*</arg_key>\s*<arg_value>(.*?)</arg_value>", re.S),
    "seed": re.compile(r"<parameter=([A-Za-z_][A-Za-z0-9_.\-]*)\s*>(.*?)</parameter>", re.S),
}

def parse_xml_calls(text: str, family: str) -> list[dict]:

    out = []
    for m in _XML_CALL[family].finditer(text):
        body = m.group(2)
        raw = {k.strip(): v for k, v in _XML_ARG[family].findall(body)}
        out.append({"name": m.group(1), "raw_args": raw})
    return out

