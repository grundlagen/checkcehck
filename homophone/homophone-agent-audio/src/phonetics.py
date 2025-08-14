def h_drop(text: str) -> str:
    # crude English h-dropping in unstressed function words
    return text.replace(" h", " ")

def yod_coalesce(ipa: str) -> str:
    return ipa.replace("t j", "tʃ").replace("d j", "dʒ")

def schwa_reduce(ipa: str) -> str:
    toks = ipa.split()
    out = []
    for i,t in enumerate(toks):
        if t == "ə" and 0 < i < len(toks)-1:
            continue
        out.append(t)
    return " ".join(out)

def apply_tricks(text: str, ipa: str):
    outs = [(text, ipa, "base")]
    v1 = (h_drop(text), ipa, "h-drop")
    if v1[0] != text: outs.append(v1)
    v2 = (text, yod_coalesce(ipa), "yod")
    if v2[1] != ipa: outs.append(v2)
    v3 = (text, schwa_reduce(ipa), "schwa-")
    if v3[1] != ipa: outs.append(v3)
    return outs