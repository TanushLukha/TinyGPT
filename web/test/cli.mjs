// Runs pieces of web/tinygpt.js on inputs from Python, so tests/test_web.py and
// scripts/check_web.py can compare them with the Python/PyTorch code.
//
//     node web/test/cli.mjs <command> <input.json>      -> JSON on stdout
import fs from "fs";
import path from "path";
import { GPT, Parser, Tokenizer, decouple, filterLogits, fromModel, iterTokens, normalize, parseTree,
         readSafetensors, slotItems, stream, topIntent, treeToString, wordSpans } from "../tinygpt.js";

function loadBundle(dir) {
  const config = JSON.parse(fs.readFileSync(path.join(dir, "config.json"), "utf8"));
  const tok = new Tokenizer(JSON.parse(fs.readFileSync(path.join(dir, config.tokenizer), "utf8")));
  const file = fs.readFileSync(path.join(dir, "model.safetensors"));
  const buffer = file.buffer.slice(file.byteOffset, file.byteOffset + file.byteLength);
  return { config, tok, model: new GPT(config.model_config, readSafetensors(buffer)) };
}

const finite = (a) => Array.from(a, (v) => (Number.isFinite(v) ? v : null));   // JSON has no -Infinity

const commands = {
  normalize: ({ texts }) => texts.map(normalize),

  tokenize: ({ tokenizer, texts }) => {
    const tok = new Tokenizer(JSON.parse(fs.readFileSync(tokenizer, "utf8")));
    return texts.map((t) => tok.encode(t));
  },

  logits: ({ bundle, ids }) => {                  // the logits after every position, fed one by one
    const { model } = loadBundle(bundle);
    const cache = model.newCache();
    return ids.map((t) => Array.from(model.step(t, cache)));
  },

  greedy: ({ bundle, ids, n, eot }) => {
    const { model } = loadBundle(bundle);
    return Array.from(iterTokens(model, ids, n, { temperature: 0, repetitionPenalty: 1, eotId: eot ?? undefined }));
  },

  filter: ({ cases }) => cases.map(({ logits, params }) => finite(filterLogits(Float32Array.from(logits), params))),

  trees: ({ parses }) => parses.map((p) => {
    try {
      const tree = parseTree(p);
      return { decoupled: treeToString(decouple(tree)), slots: slotItems(tree), intent: topIntent(p),
               spans: wordSpans(tree), upper: fromModel(p.toLowerCase()) };
    } catch (e) {
      return { error: true };
    }
  }),

  parse: ({ bundle, texts }) => {
    const { model, tok, config } = loadBundle(bundle);
    const parser = new Parser(model, tok, config.task);
    return texts.map((t) => parser.parse(t));
  },

  stream: ({ bundle, prompt, params }) => {
    const { model, tok } = loadBundle(bundle);
    const events = Array.from(stream(model, tok, prompt, params));
    return { text: events.filter((e) => e.text !== undefined).map((e) => e.text).join(""), done: events.at(-1).done };
  },

  speed: ({ bundle, n }) => {                     // milliseconds per generated token, greedy
    const { model, tok } = loadBundle(bundle);
    const ids = [tok.eotId, ...tok.encode("भारत की राजधानी")];
    const t0 = performance.now();
    let count = 0;
    for (const _ of iterTokens(model, ids, n, { temperature: 0, repetitionPenalty: 1 })) count += 1;
    return { msPerToken: (performance.now() - t0) / count, tokens: count };
  },
};

const [command, input] = process.argv.slice(2);
process.stdout.write(JSON.stringify(commands[command](JSON.parse(fs.readFileSync(input, "utf8")))));
