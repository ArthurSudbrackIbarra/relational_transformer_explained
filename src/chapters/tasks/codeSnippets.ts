/**
 * Verbatim excerpts of the paper-era official code (snap-stanford/relational-transformer @2a899c8),
 * GENERATED from the local read-only copy by copying the exact line ranges. Do not edit by hand.
 */

export interface Snippet {
  file: string;
  start: number;
  end: number;
  code: string;
}

export const TASKS_FORECAST: Snippet = {
  file: "rt/tasks.py",
  start: 1,
  end: 28,
  code: "# tuples are (database, table, target column, leakage columns)\n\nforecast_clf_tasks = [\n    (\"rel-amazon\", \"user-churn\", \"churn\", []),\n    (\"rel-hm\", \"user-churn\", \"churn\", []),\n    (\"rel-stack\", \"user-badge\", \"WillGetBadge\", []),\n    (\"rel-amazon\", \"item-churn\", \"churn\", []),\n    (\"rel-stack\", \"user-engagement\", \"contribution\", []),\n    (\"rel-avito\", \"user-visits\", \"num_click\", []),\n    (\"rel-avito\", \"user-clicks\", \"num_click\", []),\n    (\"rel-event\", \"user-ignore\", \"target\", []),\n    (\"rel-trial\", \"study-outcome\", \"outcome\", []),\n    (\"rel-f1\", \"driver-dnf\", \"did_not_finish\", []),\n    (\"rel-event\", \"user-repeat\", \"target\", []),\n    (\"rel-f1\", \"driver-top3\", \"qualifying\", []),\n]\n\nforecast_reg_tasks = [\n    (\"rel-hm\", \"item-sales\", \"sales\", []),\n    (\"rel-amazon\", \"user-ltv\", \"ltv\", []),\n    (\"rel-amazon\", \"item-ltv\", \"ltv\", []),\n    (\"rel-stack\", \"post-votes\", \"popularity\", []),\n    (\"rel-trial\", \"site-success\", \"success_rate\", []),\n    (\"rel-trial\", \"study-adverse\", \"num_of_adverse_events\", []),\n    (\"rel-event\", \"user-attendance\", \"target\", []),\n    (\"rel-f1\", \"driver-position\", \"position\", []),\n    (\"rel-avito\", \"ad-ctr\", \"num_click\", []),\n]",
};

export const TASKS_AUTOCOMPLETE: Snippet = {
  file: "rt/tasks.py",
  start: 66,
  end: 88,
  code: "autocomplete_reg_tasks = [\n    (\"rel-amazon\", \"review\", \"rating\", [\"review_text\", \"summary\"]),\n    (\n        \"rel-f1\",\n        \"results\",\n        \"position\",\n        [\n            \"statusId\",\n            \"positionOrder\",\n            \"points\",\n            \"laps\",\n            \"milliseconds\",\n            \"fastestLap\",\n            \"rank\",\n        ],\n    ),\n    (\"rel-f1\", \"qualifying\", \"position\", []),\n    (\"rel-trial\", \"studies\", \"enrollment\", []),\n    (\"rel-f1\", \"constructor_results\", \"points\", []),\n    (\"rel-f1\", \"constructor_standings\", \"position\", [\"wins\", \"points\"]),\n    (\"rel-hm\", \"transactions\", \"price\", []),\n    (\"rel-event\", \"users\", \"birthyear\", []),\n]",
};

export const DATA_TABLE_KEY: Snippet = {
  file: "rt/data.py",
  start: 77,
  end: 92,
  code: "            table_info_key = (\n                f\"{table_name}:Db\"\n                if f\"{table_name}:Db\" in table_info\n                else f\"{table_name}:{split}\"\n            )\n            info = table_info[table_info_key]\n            node_idx_offset = info[\"node_idx_offset\"]\n            num_nodes = info[\"num_nodes\"]\n\n            target_idx = get_column_index(target_column, table_name, db_name)\n            target_column_indices.append(target_idx)\n\n            drop_indices = [\n                get_column_index(col, table_name, db_name) for col in columns_to_drop\n            ]\n            drop_column_indices.append(drop_indices)",
};

export const FLY_SEEDS: Snippet = {
  file: "rustler/src/fly.rs",
  start: 287,
  end: 296,
  code: "            for j in node_idx_offset..node_idx_offset + num_nodes {\n                let node = get_node(&datasets[i], j);\n                // skip the node if target column was removed during preprocessing\n                if node.col_name_idxs.iter().any(|&c| c == target) {\n                    items.push(Item {\n                        dataset_idx: i as i32,\n                        node_idx: j,\n                    });\n                }\n            }",
};

export const FLY_TASK_TABLE_SKIP: Snippet = {
  file: "rustler/src/fly.rs",
  start: 415,
  end: 424,
  code: "            for edge in p2f_edges.iter() {\n                // include edges to task table only if seed node belongs to the task table\n                if edge.table_name_idx != seed_node.table_name_idx && edge.table_type != ArchivedTableType::Db {\n                    continue;\n                }\n\n                // temporal constraint\n                if edge.timestamp.is_some() && seed_node.timestamp.is_some() && edge.timestamp > seed_node.timestamp {\n                    continue;\n                }",
};

export const FLY_LEAKAGE_DROP: Snippet = {
  file: "rustler/src/fly.rs",
  start: 454,
  end: 461,
  code: "            let num_cells = node.col_name_idxs.len();\n            for cell_i in 0..num_cells {\n                let col_idx: i32 = node.col_name_idxs[cell_i].into();\n                if (node.node_idx == seed_node_idx && columns_to_drop.contains(&col_idx))\n                    || (node.timestamp == seed_node.timestamp && columns_to_drop.contains(&col_idx))\n                {\n                    continue; // do not add this cell to the sequence\n                }",
};

export const PRE_TABLE_TYPE: Snippet = {
  file: "rustler/src/pre.rs",
  start: 152,
  end: 174,
  code: "        let (table_name, table_type) = if is_db_table {\n            (\n                pq_path.file_stem().unwrap().to_str().unwrap().to_string(),\n                TableType::Db,\n            )\n        } else {\n            (\n                pq_path\n                    .parent()\n                    .unwrap()\n                    .file_name()\n                    .unwrap()\n                    .to_str()\n                    .unwrap()\n                    .to_string(),\n                match pq_path.file_stem().unwrap().to_str().unwrap() {\n                    \"train\" => TableType::Train,\n                    \"val\" => TableType::Val,\n                    \"test\" => TableType::Test,\n                    _ => panic!(),\n                },\n            )\n        };",
};

export const PRE_NAME_INTERN: Snippet = {
  file: "rustler/src/pre.rs",
  start: 424,
  end: 437,
  code: "        let l = text_to_idx.len() as i32;\n        let table_name_idx = *text_to_idx\n            .entry(table.table_name.clone())\n            .or_insert_with(|| l);\n\n        for (col, col_stat) in table.df.iter().zip(&table.col_stats) {\n            let col = col.rechunk();\n\n            let col_name = format!(\"{} of {}\", col.name(), table.table_name.clone());\n            let l = text_to_idx.len() as i32;\n            let col_name_idx = *text_to_idx.entry(col_name.clone()).or_insert_with(|| {\n                column_name_to_idx.push((col_name.clone(), l));\n                l\n            });",
};

export const PRE_CAST_BOOL: Snippet = {
  file: "rustler/src/pre.rs",
  start: 208,
  end: 215,
  code: "        // amazon\n        if cli.db_name == \"rel-amazon\" {\n            if table_name == \"user-churn\" {\n                df = cast_col_to_bool(df, \"churn\").unwrap();\n            }\n            if table_name == \"item-churn\" {\n                df = cast_col_to_bool(df, \"churn\").unwrap();\n            }",
};

export const FLY_MASK: Snippet = {
  file: "rustler/src/fly.rs",
  start: 492,
  end: 496,
  code: "                let is_target = seed_node_idx == node.node_idx\n                    && node.col_name_idxs[cell_i] == target_column;\n                slices.is_targets[seq_i] = is_target;\n\n                slices.masks[seq_i] = is_target;",
};

