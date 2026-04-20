# Output json file information 

## What is in `pinecone_all_vectors.json`:

The file is a single JSON object with this shape:

```json
{
  "namespace": "default",
  "vectors": [
    {
      "id": "...",
      "values": [vector1, vector2, vector3, ...],
      "metadata": {
        "text": "...",
        "level": 0,
        "node_type": "chunk",
        "parent_id": "summary_l1_c4",
        "child_ids": [],
        "source_files": ["documents/adv_nlp/class_notes/notes.org"]
      }
    }
  ]
}
```

## Meaning of each field

### Top level

- `namespace`
  Namespace intended for Pinecone upsert. Current value: `default`.
- `vectors`
  List of all vectors to store in Pinecone.

### Per vector

- `id`
  Unique node identifier.
  Examples:
  - `chunk_notes_9d0d08f9_0180`
  - `summary_l1_c4`
- `values`
  The embedding vector for that node.
  This is the numeric representation Pinecone will index for similarity search.
- `metadata`
  Human-readable information about the node.

### Metadata fields

- `metadata.text`
  The actual text that was embedded.
  - If `node_type = "chunk"`, this is original source text.
  - If `node_type = "summary"` or `root`, this is generated summary text.
- `metadata.level`
  Depth in the RAPTOR tree.
  - `0` = original chunk
  - `1` = summary of chunks
  - higher values would mean summaries of summaries
- `metadata.node_type`
  Node role in the tree.
  - `chunk`
  - `summary`
  - `root`
- `metadata.parent_id`
  Parent summary node for this node.
  `null` only for the root node.
- `metadata.child_ids`
  Child node IDs.
  Empty for chunk nodes.
- `metadata.source_files`
  Source file path(s) that contributed to this node.
  For chunk nodes this is usually one file.
  For summary nodes this can include multiple files.

## How the vectors were created

### Input corpus

- Input folder:
  `documents/adv_nlp`
- Supported file types:
  `.pdf`, `.doc`, `.docx`, `.txt`, `.org`

### Models used

- Summary model:
  `Qwen/Qwen2.5-1.5B-Instruct`
- Embedding model:
  `nomic-ai/nomic-embed-text-v1.5`

### Processing steps

1. Text was extracted from every supported file under `documents/adv_nlp`.
2. Each document was split into sentence-aware chunks.
3. Chunking settings:
   - chunk size: `100` tokens
   - overlap: `20` tokens
4. Each chunk was embedded with `nomic-ai/nomic-embed-text-v1.5`.
5. Chunk embeddings were clustered with:
   - `UMAP` for dimensionality reduction
   - `Gaussian Mixture Model` for clustering
6. Each cluster was summarized into a parent node using `Qwen/Qwen2.5-1.5B-Instruct`.
7. Those summaries were embedded again and written as additional vectors.
8. The final export contains both:
   - chunk vectors
   - summary vectors

## Current corpus stats

From `pinecone_stats.json`:

- input supported files: `130`
- exported source files: `130`
- total vectors: `45752`
- chunk nodes: `45740`
- summary nodes: `11`
- root nodes: `1`

### Levels present

- level `0`: `45740` chunk nodes
- level `1`: `12` summary/root nodes (the clusters didn't find similarity between the 12 lvl 1 summaries to further
                    group them into multiple clusters)