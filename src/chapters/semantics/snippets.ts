/**
 * Verbatim excerpts of the paper-era official code (snap-stanford/relational-transformer @2a899c8),
 * extracted programmatically with their exact line numbers. Do not edit by hand.
 */

/** rt/embed.py:12-31. TextEmbedder: a SentenceTransformer, encode() with its defaults (spec §3.6). */
export const EMBED_TEXT_EMBEDDER_12_31 = "class TextEmbedder:\n    def __init__(self, batch_size, embedding_model, device_type):\n        self.model = SentenceTransformer(\n            f\"sentence-transformers/{embedding_model}\",\n            device=device_type,\n            model_kwargs={\n                \"dtype\": torch.bfloat16 if device_type == \"cuda\" else torch.float32,\n            },\n        )\n        self.model = torch.compile(self.model)\n        self.batch_size = batch_size\n\n    def __call__(self, text_list, device=None):\n        return self.model.encode(\n            text_list,\n            batch_size=self.batch_size,\n            convert_to_numpy=True,\n            show_progress_bar=True,\n            device=device,\n        )";

/** rt/embed.py:65-72. Every string of text.json embedded once, stored as a raw [N, 384] bf16 file (spec §3.6). */
export const EMBED_MAIN_65_72 = "    emb_list = text_embedder(\n        text_list,\n        device=device\n    )\n\n    emb_path = f\"{os.environ['HOME']}/scratch/pre/{dataset_name}/text_emb_{embedding_model}.bin\"\n    emb = np.stack(emb_list).astype(bfloat16)\n    emb.tofile(emb_path)";

/** rustler/src/pre.rs:424-437. Interning the table name and the phrase "<col> of <table>" (spec §3.6). */
export const PRE_INTERN_424_437 = "        let l = text_to_idx.len() as i32;\n        let table_name_idx = *text_to_idx\n            .entry(table.table_name.clone())\n            .or_insert_with(|| l);\n\n        for (col, col_stat) in table.df.iter().zip(&table.col_stats) {\n            let col = col.rechunk();\n\n            let col_name = format!(\"{} of {}\", col.name(), table.table_name.clone());\n            let l = text_to_idx.len() as i32;\n            let col_name_idx = *text_to_idx.entry(col_name.clone()).or_insert_with(|| {\n                column_name_to_idx.push((col_name.clone(), l));\n                l\n            });";

/** rustler/src/pre.rs:439-442. A primary-key column produces no cell (spec §3.1). */
export const PRE_PK_439_442 = "            if col.name() == table.pcol_name.as_deref().unwrap_or(\"\") {\n                pbar.inc(col.len() as u64);\n                continue;\n            }";

/** rustler/src/pre.rs:595-605. A string cell stores the index of its raw value (spec §3.6). */
export const PRE_TEXT_595_605 = "                    AnyValue::String(val) => {\n                        let l = text_to_idx.len() as i32;\n                        let text_idx = *text_to_idx.entry(val.to_string()).or_insert_with(|| l);\n                        node.boolean_values.push(0.0);\n                        node.number_values.push(0.0);\n                        node.text_values.push(text_idx);\n                        node.datetime_values.push(0.0);\n                        node.sem_types.push(SemType::Text);\n                        node.col_name_idxs.push(col_name_idx);\n                        node.class_value_idx.push(text_idx);\n                    }";

/** rt/data.py:31-43. Resolving a target column through its phrase (spec §3.6). */
export const DATA_COLUMN_INDEX_31_43 = "def get_column_index(column_name: str, table_name: str, db_name: str) -> int:\n    \"\"\"\n    Get the index of a column in the text embeddings for a given dataset.\n    \"\"\"\n    column_index = _load_column_index(db_name)\n    target = f\"{column_name} of {table_name}\"\n\n    if target not in column_index:\n        raise ValueError(\n            f'Column \"{target}\" not found in column_index.json for dataset {db_name}.'\n        )\n\n    return column_index[target]";

/** rustler/src/fly.rs:469-486. The sampler copies the phrase and value embeddings into the batch (spec §5.1). */
export const FLY_COPY_469_486 = "                slices.table_name_idxs[seq_i] = node.table_name_idx.into();\n                slices.col_name_idxs[seq_i] = node.col_name_idxs[cell_i].into();\n                slices.class_value_idxs[seq_i] = node.class_value_idx[cell_i].into();\n                slices.col_name_values[seq_i * self.d_text..(seq_i + 1) * self.d_text]\n                    .copy_from_slice(get_text_emb(\n                        dataset,\n                        slices.col_name_idxs[seq_i],\n                        self.d_text,\n                    ));\n\n                let s = node.sem_types[cell_i].clone() as i32;\n                slices.sem_types[seq_i] = s;\n\n                slices.number_values[seq_i] = bf16::from_f32(node.number_values[cell_i].into());\n\n                let text_idx: i32 = node.text_values[cell_i].into();\n                slices.text_values[seq_i * self.d_text..(seq_i + 1) * self.d_text]\n                    .copy_from_slice(get_text_emb(dataset, text_idx, self.d_text));";

/** rustler/src/fly.rs:528-532. get_text_emb: a row lookup in the memory-mapped table (spec §3.6). */
export const FLY_GET_TEXT_EMB_528_532 = "fn get_text_emb(dataset: &Dataset, idx: i32, d_text: usize) -> &[bf16] {\n    let (pref, text_emb, suf) = unsafe { dataset.text_mmap.align_to::<bf16>() };\n    assert!(pref.is_empty() && suf.is_empty());\n    &text_emb[(idx as usize) * d_text..(idx as usize + 1) * d_text]\n}";

/** rt/model.py:115-123. The learned projections 384 -> d_model (spec §5.2). */
export const MODEL_ENC_DICT_115_123 = "        self.enc_dict = nn.ModuleDict(\n            {\n                \"number\": nn.Linear(1, d_model, bias=True),\n                \"text\": nn.Linear(d_text, d_model, bias=True),\n                \"datetime\": nn.Linear(1, d_model, bias=True),\n                \"col_name\": nn.Linear(d_text, d_model, bias=True),\n                \"boolean\": nn.Linear(1, d_model, bias=True),\n            }\n        )";

/** rt/model.py:208-224. Input embedding: column-name term plus gated value terms (spec §5.2). */
export const MODEL_EMBED_208_224 = "        x = 0\n        x = x + (\n            self.norm_dict[\"col_name\"](\n                self.enc_dict[\"col_name\"](batch[\"col_name_values\"])\n            )\n            * (~is_padding)[..., None]\n        )\n\n        for i, t in enumerate([\"number\", \"text\", \"datetime\", \"boolean\"]):\n            x = x + (\n                self.norm_dict[t](self.enc_dict[t](batch[t + \"_values\"]))\n                * ((batch[\"sem_types\"] == i) & ~batch[\"masks\"] & ~is_padding)[..., None]\n            )\n            x = x + (\n                self.mask_embs[t]\n                * ((batch[\"sem_types\"] == i) & batch[\"masks\"] & ~is_padding)[..., None]\n            )";
