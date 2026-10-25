import pytest
import torch

from tinygpt.bpe import BPETokenizer
from tinygpt.models import GPT, GPTConfig
from tinygpt.top import (Example, decoupled, example_ids, fewshot_prompt_ids, from_model, load, make_batch,
                         match_key, parse_tree, predict, prompt_ids, score, scored_form, slot_items, to_model,
                         to_string, top_intent)

FULL = "[IN:CREATE_ALARM [SL:DATE_TIME do ghante ke liye ] alarm set kardo ]"
NESTED = ("[IN:GET_ESTIMATED_ARRIVAL agar mai [SL:DATE_TIME_DEPARTURE raat 9 baje tak ] "
          "[SL:SOURCE [IN:GET_LOCATION [SL:LOCATION_USER idhar ] ] ] se nikal jau toh kya mai "
          "[SL:DESTINATION Taipei ] pahunch jaunga ? ]")


def test_tree_round_trip_and_decoupling():
    assert to_string(parse_tree(FULL)) == FULL
    assert decoupled(FULL) == "[IN:CREATE_ALARM [SL:DATE_TIME do ghante ke liye ] ]"
    assert decoupled(NESTED) == ("[IN:GET_ESTIMATED_ARRIVAL [SL:DATE_TIME_DEPARTURE raat 9 baje tak ] "
                                 "[SL:SOURCE [IN:GET_LOCATION [SL:LOCATION_USER idhar ] ] ] [SL:DESTINATION Taipei ] ]")
    assert decoupled("[IN:GET_MESSAGE Please mere chat requests ko retrieve kare ]") == "[IN:GET_MESSAGE ]"


@pytest.mark.parametrize("bad", [
    "[IN:A [SL:B x ]",                  # unclosed
    "[IN:A x ] ]",                      # extra ]
    "x [IN:A ]",                        # word outside the tree
    "[SL:B x ]",                        # root must be an intent
    "[IN:A [IN:B ] ]",                  # intent directly inside an intent
    "[IN:A [SL:B [SL:C x ] ] ]",        # slot directly inside a slot
    "[IN:A ] [IN:B ]",                  # two roots
    "[XX:A ]", "[IN: ]", "",
])
def test_invalid_trees_are_rejected(bad):
    with pytest.raises(ValueError):
        parse_tree(bad)


def test_slots_and_intent():
    tree = parse_tree(decoupled(NESTED))
    assert slot_items(tree) == [("SL:DATE_TIME_DEPARTURE", "raat 9 baje tak"), ("SL:SOURCE", "idhar"),
                                ("SL:LOCATION_USER", "idhar"), ("SL:DESTINATION", "Taipei")]
    assert top_intent(NESTED) == "IN:GET_ESTIMATED_ARRIVAL"
    assert top_intent("hello") is None


def test_score():
    gold = [decoupled(FULL), decoupled(NESTED)]
    assert score(gold, gold) == {"n": 2, "exact_match": 1.0, "intent_acc": 1.0, "slot_p": 1.0, "slot_r": 1.0,
                                 "slot_f1": 1.0, "valid": 1.0}
    preds = ["[IN:CREATE_ALARM [SL:DATE_TIME do ghante ] ]",            # valid tree, one wrong slot
             "[IN:GET_ESTIMATED_ARRIVAL [SL:DESTINATION Taipei ]"]     # right intent, not a tree
    s = score(preds, gold)
    assert s["exact_match"] == 0 and s["intent_acc"] == 1.0 and s["valid"] == 0.5
    assert s["slot_p"] == 0.0 and s["slot_r"] == 0.0                  # an invalid tree has no slots


def test_label_format_round_trip():
    p = decoupled(NESTED)
    low = to_model(p)
    assert low.startswith("[in:get_estimated_arrival [sl:date_time_departure raat 9 baje tak ]")
    assert "Taipei" in low                                     # words keep their case
    assert from_model(low) == p and to_model(p, "upper") == p
    assert match_key("Kal ka  WEATHER?") == match_key("kal ka weather")


@pytest.fixture(scope="module")
def tok():
    text = ["do ghante ke liye alarm set kardo => [in:create_alarm [sl:date_time do ghante ke liye ] ]",
            "kal ka weather kaisa hoga => [in:get_weather [sl:date_time kal ] ]"]
    return BPETokenizer.train(text * 20, vocab_size=320)


def test_example_ids_and_masked_batch(tok):
    ex = Example("do ghante ke liye alarm set kardo", decoupled(FULL), "alarm")
    ids, n_prompt = example_ids(tok, ex)
    assert ids[0] == tok.eot_id and ids[-1] == tok.eot_id and ids[:n_prompt] == prompt_ids(tok, ex.query)
    assert tok.decode(ids[1:-1]) == ex.query + " => " + to_model(ex.parse)
    short = example_ids(tok, Example("kal ka weather", "[IN:GET_WEATHER [SL:DATE_TIME kal ] ]", "weather"))
    x, y = make_batch([(ids, n_prompt), short], pad_id=tok.eot_id)
    assert x.shape == y.shape == (2, len(ids) - 1)
    assert x[0].tolist() == ids[:-1]
    assert (y[0, :n_prompt - 1] == -100).all()                       # no loss on the request
    assert y[0, n_prompt - 1:].tolist() == ids[n_prompt:]            # loss on the parse and the end
    n_short = len(short[0]) - 1
    assert (y[1, n_short:] == -100).all()                            # no loss on padding


def test_fewshot_prompt(tok):
    demo = Example("kal ka weather", "[IN:GET_WEATHER [SL:DATE_TIME kal ] ]", "weather")
    ids = fewshot_prompt_ids(tok, [demo], "aaj ka weather")
    assert tok.decode(ids[1:]) == "kal ka weather => [in:get_weather [sl:date_time kal ] ]\naaj ka weather =>"


def test_predict_batches_like_one_by_one(tok):
    torch.manual_seed(0)
    m = GPT(GPTConfig(vocab_size=tok.vocab_size, block_size=64, d_model=32, n_layers=1, n_heads=2)).eval()
    prompts = [prompt_ids(tok, q) for q in ["kal ka weather", "do ghante ke liye alarm", "kal ka weather kaisa"]]
    together = predict(m, tok, prompts, "cpu", batch_size=2, max_new_tokens=12)
    alone = [predict(m, tok, [p], "cpu", max_new_tokens=12)[0] for p in prompts]
    assert together == alone and all(isinstance(t, str) for t in together)


def test_load_reads_the_tsv(tmp_path):
    header = "en_query\tcs_query\ten_parse\tcs_parse\tdomain\n"
    row = "set alarm for 2 hours\tdo  ghante ke liye alarm set kardo\t[IN:CREATE_ALARM ]\t" + FULL + "\talarm\n"
    for split in ("train", "val", "test"):
        (tmp_path / f"{split}.tsv").write_text(header + row, encoding="utf-8")
    [ex] = load("val", tmp_path)
    assert ex == Example("do ghante ke liye alarm set kardo", "[IN:CREATE_ALARM [SL:DATE_TIME do ghante ke liye ] ]",
                         "alarm", FULL)


def test_annotated_target_is_scored_on_the_decoupled_tree(tok):
    ex = Example("do ghante ke liye alarm set kardo", decoupled(FULL), "alarm", FULL)
    ids, n = example_ids(tok, ex, target="annotated")
    assert tok.decode(ids[n:-1]) == " " + to_model(FULL)               # the request, with brackets
    assert scored_form(FULL) == ex.parse
    assert scored_form("[IN:CREATE_ALARM alarm") == "[IN:CREATE_ALARM alarm"   # not a tree: kept, scored invalid


class Scripted:
    """Stands in for a model: always continues with the same tokens; records how many it was asked for."""
    def __init__(self, tokens, block_size=64):
        self.tokens, self.cfg, self.asked = tokens, type("cfg", (), {"block_size": block_size}), []

    def iter_tokens(self, x, n, temperature=0):
        self.asked.append(n)
        for t in self.tokens[:n]:
            yield torch.full((x.size(0), 1), t)


def test_predict_stays_inside_the_window_and_takes_the_first_line(tok):
    nl, eot = tok.encode_ordinary("\n")[0], tok.eot_id
    words = tok.encode_ordinary(" kal ka weather")
    m = Scripted([nl] + words + [nl] + words + [eot], block_size=64)
    assert predict(m, tok, [[eot] * 60], "cpu", max_new_tokens=96, stop="\n") == ["kal ka weather"]
    assert m.asked == [4]                                           # 64-token window, 60-token prompt
