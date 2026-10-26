"""The Gradio demo (M6). One function builds it; the caller supplies the two model calls:

    generate(prompt, GenerateParams) -> iterator of Piece ... Done
    parse(list of requests)          -> list of ParseResult

The local server passes calls that go through its worker threads (api.py); the Hugging Face Space
passes the engines' own methods wrapped in @spaces.GPU (space/app.py). Same page, same code.
"""
from __future__ import annotations

from typing import Callable, Iterator

import gradio as gr

from .engine import Done, GenerateParams, ParseResult, Piece

GITHUB = "https://github.com/TanushLukha/TinyGPT"
# The first one is the default. A factual opener ("भारत की राजधानी") stays among the examples: the
# note above the box says what it will show — fluent text, invented facts (docs/results/m6_facts.md).
PROMPTS = ["हिन्दी भाषा", "भारत की राजधानी", "महात्मा गांधी का जन्म", "yaar kal movie dekhne", "mujhe kal subah 7 baje"]
WRITE_NOTE = (
    "TinyGPT continues text the way its training text — Hindi Wikipedia and Hinglish — reads. At 13.8M "
    "parameters it learned the language, not the facts: asked for the capitals of 20 states and countries, "
    "it ranks the right city first for only 2 (chance level), the two its training text states most often. "
    "Expect fluent sentences with invented names, dates and places.")
REQUESTS = ["kal subah 7 baje ka alarm laga do", "mujhe Delhi ka weather batao",
            "agle hafte mummy ko call karne ka reminder set karo", "Arijit Singh ke gaane bajao",
            "office se ghar pahunchne mein kitna time lagega"]


def theme() -> gr.themes.Base:
    return gr.themes.Soft(primary_hue="orange", secondary_hue="blue")


def params_from_ui(max_new_tokens, temperature, top_k, top_p, min_p, repetition_penalty, seed) -> GenerateParams:
    """Slider values -> GenerateParams; a knob at its 'off' end is None, as the API expects."""
    return GenerateParams(
        max_new_tokens=int(max_new_tokens), temperature=float(temperature), top_k=int(top_k) or None,
        top_p=None if float(top_p) >= 1.0 else float(top_p), min_p=float(min_p) or None,
        repetition_penalty=float(repetition_penalty),
        seed=None if seed is None or int(seed) < 0 else int(seed))


def stats_line(done: Done) -> str:
    finish = {"stop": "the model ended the text", "length": "reached the token limit",
              "cancelled": "stopped"}[done.finish_reason]
    per_token = (done.seconds - (done.first_token_seconds or 0)) / max(done.tokens - 1, 1) * 1000
    return (f"{done.tokens} tokens in {done.seconds:.2f} s ({per_token:.1f} ms per token after the first) · "
            f"{finish} · seed {done.seed}")


def parse_outputs(r: ParseResult) -> tuple[list[tuple[str, str | None]], str, str]:
    """-> (highlighted words, summary markdown, the tree as written) for one result."""
    if r.error:
        return [], f"⚠️ {r.error}", ""
    if not r.valid:
        return [], "The model did not write a well-formed tree this time.", r.written
    rows = "\n".join(f"| `{s['label']}` | {s['text']} |" for s in r.slots) or "| — | (no slots) |"
    summary = f"**Intent:** `{r.intent}`\n\n| slot | words |\n|---|---|\n{rows}"
    return [(s["text"], s["label"]) for s in r.spans], summary, r.written


def about(lm: dict, parser: dict) -> str:
    """The About tab, from the two bundles' config.json."""
    bpb = lm.get("scores", {}).get("val_bpb", {})
    test = parser.get("scores", {}).get("test", {})
    return f"""
**TinyGPT** is a decoder-only Transformer (Llama recipe: RoPE, RMSNorm, SwiGLU), {lm['params'] / 1e6:.1f}M
parameters, trained from scratch on 18M tokens of Hindi Wikipedia and Hinglish on a MacBook. Everything
is written by hand in PyTorch: the BPE tokenizer, the model, training, decoding, LoRA, and this service.

| measure | value |
|---|---|
| bits per byte on held-out text (lower is better) | {bpb.get('all', float('nan')):.3f} (Hindi {bpb.get('hindi', float('nan')):.3f}, Hinglish {bpb.get('hinglish', float('nan')):.3f}) |
| context window | {lm['model_config']['block_size']} tokens |
| capitals of 20 states and countries: right city ranked first | 2 of 20 — chance level ([details]({GITHUB}/blob/main/docs/results/m6_facts.md)) |
| parser: whole tree right on unseen test requests | {test.get('exact_match', float('nan')):.1%} |
| parser: intent right | {test.get('intent_acc', float('nan')):.1%} |

The parser is its twin — pretrained the same way but without Hinglish-TOP's test requests — fine-tuned
on Hinglish-TOP (M5). It writes the request back with brackets around the slots. Its most common
mistake is a slot missing, extra or mislabelled; about one mistake in five puts words in a slot that
the request doesn't contain (*494 northbound* → *478 northbound*).
Code, experiments and write-ups: [{GITHUB}]({GITHUB}).
"""


def build_ui(generate: Callable[[str, GenerateParams], Iterator[Piece | Done]],
             parse: Callable[[list[str]], list[ParseResult]],
             lm_config: dict, parser_config: dict) -> gr.Blocks:
    def on_generate(prompt, *knobs):
        params = params_from_ui(*knobs)
        text = prompt
        yield text, "…"
        try:
            for item in generate(prompt, params):
                if isinstance(item, Piece):
                    text += item.text
                    yield text, gr.skip()
                else:
                    yield text, stats_line(item)
        except Exception as e:                          # busy, or the model failed: say so
            raise gr.Error(str(e)) from e

    def on_parse(texts: list[str]):                     # batch=True: lists in, one list per output out
        try:
            results = parse(list(texts))
        except Exception as e:
            raise gr.Error(str(e)) from e
        outs = [parse_outputs(r) for r in results]
        return [o[0] for o in outs], [o[1] for o in outs], [o[2] for o in outs]

    with gr.Blocks(title="TinyGPT — Hindi & Hinglish") as demo:
        gr.Markdown("# TinyGPT\nA small GPT for **Hindi** and **Hinglish**, built from scratch — "
                    f"tokenizer, model, training, fine-tuning and this demo. [Code and write-ups]({GITHUB})")
        with gr.Tab("Write"):
            gr.Markdown(WRITE_NOTE)
            with gr.Row():
                with gr.Column(scale=3):
                    prompt = gr.Textbox(PROMPTS[0], label="Start of the text", lines=2)
                    with gr.Row():
                        go = gr.Button("Generate", variant="primary")
                        stop = gr.Button("Stop")
                    output = gr.Textbox(label="Text", lines=10, buttons=["copy"])
                    stats = gr.Markdown()
                    gr.Examples([[p] for p in PROMPTS], inputs=[prompt])
                with gr.Column(scale=2):
                    with gr.Accordion("Decoding (M4)", open=True):
                        knobs = [
                            gr.Slider(8, 256, 120, step=8, label="max new tokens"),
                            gr.Slider(0, 1.5, 0.8, step=0.05, label="temperature", info="0 = greedy"),
                            gr.Slider(0, 200, 50, step=1, label="top-k", info="0 = off"),
                            gr.Slider(0.05, 1.0, 1.0, step=0.05, label="top-p", info="1 = off"),
                            gr.Slider(0, 0.5, 0.0, step=0.01, label="min-p", info="0 = off"),
                            gr.Slider(1.0, 2.0, 1.0, step=0.05, label="repetition penalty", info="1 = off"),
                            gr.Number(-1, label="seed", info="-1 = random", precision=0),
                        ]
            events = [go.click(on_generate, [prompt, *knobs], [output, stats], api_name="generate"),
                      prompt.submit(on_generate, [prompt, *knobs], [output, stats], api_visibility="private")]
            stop.click(None, cancels=events)

        with gr.Tab("Understand a request"):
            test = parser_config.get("scores", {}).get("test") or {}
            gr.Markdown(f"The fine-tuned parser from M5 finds the intent and the slots an assistant would act on. "
                        f"On requests it never saw it gets the whole tree right {test.get('exact_match', 0):.0%} "
                        f"of the time and the intent {test.get('intent_acc', 0):.0%}.")
            request = gr.Textbox(REQUESTS[0], label="A request to a voice assistant, in Hinglish")
            parse_btn = gr.Button("Parse", variant="primary")
            spans = gr.HighlightedText(label="Slots in the request, as the model wrote it", show_legend=True)
            summary = gr.Markdown()
            written = gr.Textbox(label="The tree the model wrote", buttons=["copy"])
            gr.Examples([[r] for r in REQUESTS], inputs=[request])
            parse_btn.click(on_parse, [request], [spans, summary, written], batch=True, max_batch_size=16,
                            api_name="parse")
            request.submit(on_parse, [request], [spans, summary, written], batch=True, max_batch_size=16,
                           api_visibility="private")

        with gr.Tab("About"):
            gr.Markdown(about(lm_config, parser_config))
    return demo
