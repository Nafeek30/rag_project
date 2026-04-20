# RAPTOR Indexing

This folder builds a local RAPTOR-style tree and exports one Pinecone-ready handoff file.

## Models used

- Summary model: `Qwen/Qwen2.5-1.5B-Instruct`
- Embedding model: `nomic-ai/nomic-embed-text-v1.5`

## What the pipeline does

- reads supported documents recursively from the input folder
- extracts text from each file
- splits text into sentence-aware chunks of about 100 tokens
- embeds each chunk
- groups related chunks with `UMAP + Gaussian Mixture Model`
- writes one paragraph summary for each group
- embeds those summaries
- repeats the grouping and summarizing process until further clustering is no longer useful
- exports one Pinecone-ready JSON file

## Supported input files

- `.pdf`
- `.doc`
- `.docx`
- `.txt`
- `.org`

## Install

```bash
python3 -m pip install -r indexing/requirements.txt
```

## Run

Whole folder (eg):

```bash
python3 indexing/index.py --input-dir documents/adv_nlp --output-dir indexing/output/latest
```

Single file (eg):

```bash
python3 indexing/index.py --input-dir documents/adv_nlp --file class_notes/syllabus.org --output-dir indexing/output/latest
```

Current defaults:

- chunk size: `100` tokens
- chunk overlap: `20` tokens
- summary size: `180` generated tokens

## Output

Database upload file:

- `indexing/output/latest/pinecone_all_vectors.json`

stats file:

- `indexing/output/latest/pinecone_stats.json`

Its shape is:

- `namespace`
- `vectors`

Each vector contains:

- `id`
- `values`
- `metadata.text`
- `metadata.level`
- `metadata.node_type`
- `metadata.parent_id`
- `metadata.child_ids`
- `metadata.source_files`

The stats file reports:

- how many supported input files were discovered
- how many files failed extraction
- how many source files actually made it into the export
- how many chunk and summary nodes were created
- how many nodes exist at each tree level
- how many chunk nodes came from each source file
