// Where the page downloads the two model bundles from. scripts/deploy_hf.py replaces this file in
// the Space with the Hugging Face model repo at a pinned commit; for local testing
// (python -m http.server, then http://localhost:8000/web/) it points at the exported models/.
export const MODEL_BASE = "../models/";
export const MODEL_PAGE = "https://huggingface.co/tanushlukha/tinygpt";
