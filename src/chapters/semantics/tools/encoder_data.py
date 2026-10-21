"""Offline facts about RT's frozen text encoder, for chapter 5 ("semantics").

Writes src/chapters/semantics/encoderData.json:
  * the WordPiece tokens the all-MiniLM-L12-v2 tokenizer produces for every string in
    public/data/embeddings/minilm-l12-v2.json (the strings python/embed.py embedded);
  * the encoder's size (layers, hidden, heads, vocab, parameter count, max_seq_length);
  * a check that the stored vectors equal mean pooling over ALL wordpieces ([CLS] and [SEP]
    included, i.e. the attention mask) followed by L2 normalisation, which is what the
    sentence-transformers pipeline Transformer -> Pooling(mean) -> Normalize does (spec §3.6).

Nothing here is used by the simulation itself; it only feeds the chapter's explanations.

Usage: python/.venv/bin/python -I src/chapters/semantics/tools/encoder_data.py
"""
import json
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[4]
meta = json.loads((ROOT / 'public/data/embeddings/minilm-l12-v2.json').read_text(encoding='utf-8'))
strings = meta['strings']
emb = np.fromfile(ROOT / 'public/data/embeddings/minilm-l12-v2.f32', dtype='<f4').reshape(-1, meta['dim'])

model = SentenceTransformer(meta['model'], device='cpu')
tok = model.tokenizer
bert = model[0].auto_model

wordpieces = {}
worst = 0.0
worst_without_special = 1e9
for i, s in enumerate(strings):
    enc = tok([s], return_tensors='pt', truncation=True, max_length=model.max_seq_length)
    wordpieces[s] = tok.convert_ids_to_tokens(enc['input_ids'][0].tolist())
    with torch.no_grad():
        h = bert(**enc).last_hidden_state[0]
    v = h.mean(0)
    v = (v / v.norm()).numpy()
    worst = max(worst, float(np.abs(v - emb[i]).max()))
    if h.shape[0] > 2:
        w = h[1:-1].mean(0)
        w = (w / w.norm()).numpy()
        worst_without_special = min(worst_without_special, float(np.abs(w - emb[i]).max()))

out = {
    'model': meta['model'],
    'maxSeqLength': model.max_seq_length,
    'parameters': int(sum(p.numel() for p in model.parameters())),
    'layers': bert.config.num_hidden_layers,
    'hidden': bert.config.hidden_size,
    'heads': bert.config.num_attention_heads,
    'vocab': bert.config.vocab_size,
    'lowercase': bool(getattr(tok, 'do_lower_case', True)),
    'modules': [type(m).__name__ for m in model],
    'meanPoolCheck': {
        'maxAbsDiffAllWordpieces': worst,
        'minMaxAbsDiffWithoutClsSep': worst_without_special,
    },
    'maxWordpieces': max(len(t) for t in wordpieces.values()),
    'wordpieces': wordpieces,
}
path = Path(__file__).resolve().parents[1] / 'encoderData.json'
path.write_text(json.dumps(out, ensure_ascii=False, separators=(',', ':')) + '\n', encoding='utf-8')
print({k: v for k, v in out.items() if k != 'wordpieces'}, path)
