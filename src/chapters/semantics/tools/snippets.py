"""Regenerate ../snippets.ts: verbatim excerpts of the paper-era official code with exact line numbers.

UP must point at the local copy of snap-stanford/relational-transformer @2a899c8 (BUILD_PLAN §1).
Usage: python3 src/chapters/semantics/tools/snippets.py
"""
import json
from pathlib import Path
UP = Path('/tmp/claude-1004/-home-ibarra-Documents-Mestrado-Relational-Transformer/155ecce6-1f67-4870-9609-2c9740593feb/scratchpad/upstream/rt-paper-era')
OUT = Path(__file__).resolve().parents[1] / 'snippets.ts'
specs = [
  ('EMBED_TEXT_EMBEDDER_12_31', 'rt/embed.py', 12, 31, 'TextEmbedder: a SentenceTransformer, encode() with its defaults (spec §3.6).'),
  ('EMBED_MAIN_65_72', 'rt/embed.py', 65, 72, 'Every string of text.json embedded once, stored as a raw [N, 384] bf16 file (spec §3.6).'),
  ('PRE_INTERN_424_437', 'rustler/src/pre.rs', 424, 437, 'Interning the table name and the phrase "<col> of <table>" (spec §3.6).'),
  ('PRE_PK_439_442', 'rustler/src/pre.rs', 439, 442, 'A primary-key column produces no cell (spec §3.1).'),
  ('PRE_TEXT_595_605', 'rustler/src/pre.rs', 595, 605, 'A string cell stores the index of its raw value (spec §3.6).'),
  ('DATA_COLUMN_INDEX_31_43', 'rt/data.py', 31, 43, 'Resolving a target column through its phrase (spec §3.6).'),
  ('FLY_COPY_469_486', 'rustler/src/fly.rs', 469, 486, 'The sampler copies the phrase and value embeddings into the batch (spec §5.1).'),
  ('FLY_GET_TEXT_EMB_528_532', 'rustler/src/fly.rs', 528, 532, 'get_text_emb: a row lookup in the memory-mapped table (spec §3.6).'),
  ('MODEL_ENC_DICT_115_123', 'rt/model.py', 115, 123, 'The learned projections 384 -> d_model (spec §5.2).'),
  ('MODEL_EMBED_208_224', 'rt/model.py', 208, 224, 'Input embedding: column-name term plus gated value terms (spec §5.2).'),
]
lines = ['/**', ' * Verbatim excerpts of the paper-era official code (snap-stanford/relational-transformer @2a899c8),',
         ' * extracted programmatically with their exact line numbers. Do not edit by hand.', ' */', '']
for name, f, a, b, doc in specs:
    src = (UP / f).read_text().split('\n')
    code = '\n'.join(src[a - 1:b])
    lines.append(f'/** {f}:{a}-{b}. {doc} */')
    lines.append(f'export const {name} = {json.dumps(code, ensure_ascii=False)};')
    lines.append('')
OUT.write_text('\n'.join(lines))
print('wrote', OUT)
