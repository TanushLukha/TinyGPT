import unicodedata

from tinygpt.data import _group, _pick_hinglish
from tinygpt.text import is_val, normalize, script_shares, strip_wiki_tail


def test_normalize_nfc_and_whitespace():
    decomposed = "\u0915\u093c"                                   # KA + NUKTA
    assert normalize(decomposed) == unicodedata.normalize("NFC", decomposed)
    assert normalize("  a\u00a0\u00a0b \r\n\r\n\r\n\nc\t ") == "a b\n\nc"
    assert normalize("\ufeffशब्द\u00ad\u200b") == "शब्द"     # BOM, soft hyphen, zero-width space go
    assert normalize("मल्\u200dलों") == "मल्\u200dलों"         # zero-width joiner stays


def test_script_shares():
    hi, la = script_shares("भारत India")
    assert abs(hi - 4 / 9) < 1e-9 and abs(la - 5 / 9) < 1e-9
    assert script_shares("123 !!") == (0.0, 0.0)


def test_strip_wiki_tail():
    text = "पहला अनुच्छेद\nदूसरा\nसन्दर्भ\n1. कोई किताब"
    assert strip_wiki_tail(text) == "पहला अनुच्छेद\nदूसरा"
    assert strip_wiki_tail("कोई सन्दर्भ नहीं") == "कोई सन्दर्भ नहीं"   # only whole heading lines


def test_is_val_deterministic():
    docs = [f"document {i}" for i in range(4000)]
    picks = [is_val(d, 20) for d in docs]
    assert picks == [is_val(d, 20) for d in docs]
    assert 40 < sum(picks) < 130                              # about 2%


def test_group_dedups_and_chunks():
    out = list(_group(iter(["a", "b", "a", "c", "d", "e"]), per_doc=2))
    assert out == ["a\nb", "c\nd", "e"]


def test_pick_hinglish():
    assert _pick_hinglish({"translation": {"en": "hi", "hi_ng": "kaise ho"}}) == "kaise ho"
    assert _pick_hinglish({"hinglish": "theek hai"}) == "theek hai"
    assert _pick_hinglish({"text": "?"}) is None
