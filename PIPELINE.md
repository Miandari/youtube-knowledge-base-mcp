# YouTube Knowledge Base MCP - Full RAG Pipeline

## INGESTION PIPELINE (YouTube URL → Stored Chunks)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  1. VIDEO ACQUISITION                                                        │
│  ────────────────────                                                        │
│  Tool: yt-dlp                                                                │
│  Input: YouTube URL                                                          │
│  Output: Video metadata + Subtitle file (.vtt)                               │
│                                                                              │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐               │
│  │ YouTube URL  │ ───► │   yt-dlp     │ ───► │  Subtitles   │               │
│  │              │      │  (download)  │      │  + Metadata  │               │
│  └──────────────┘      └──────────────┘      └──────────────┘               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  2. TRANSCRIPT PARSING                                                       │
│  ─────────────────────                                                       │
│  Tool: webvtt-py                                                             │
│  Technique: VTT timestamp extraction                                         │
│  Output: List of {text, start_time, end_time} segments                       │
│                                                                              │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐               │
│  │  .vtt file   │ ───► │  VTT Parser  │ ───► │  Timestamped │               │
│  │              │      │              │      │   Segments   │               │
│  └──────────────┘      └──────────────┘      └──────────────┘               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  3. SEMANTIC CHUNKING                                                        │
│  ────────────────────                                                        │
│  Technique: Embedding-based topic shift detection                            │
│  Model: all-MiniLM-L6-v2 (local, FREE) - NOT voyage-3-large                 │
│  Config: ~500 chars/chunk, 150 char overlap                                  │
│                                                                              │
│  Cost Optimization: Use cheap local model for topic detection only.          │
│  Voyage/OpenAI only used for final chunk embeddings (Step 5).               │
│                                                                              │
│  How it works:                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ 1. Split transcript into sentences                                   │    │
│  │ 2. Add context window (1 sentence before/after each)                 │    │
│  │ 3. Embed each sentence with all-MiniLM-L6-v2 (local, ~80MB)         │    │
│  │ 4. Calculate cosine distance between consecutive embeddings          │    │
│  │ 5. Find breakpoints at 80th percentile (topic shifts)                │    │
│  │ 6. Group sentences between breakpoints into chunks                   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  Fallback: If sentence-transformers not installed, uses sentence-boundary   │
│                                                                              │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐               │
│  │  Timestamped │ ───► │  Semantic    │ ───► │   Coherent   │               │
│  │   Segments   │      │  Chunker     │      │    Chunks    │               │
│  └──────────────┘      └──────────────┘      └──────────────┘               │
│                                                                              │
│  Preserves: Timestamp boundaries for each chunk                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  4. CONTEXTUAL RETRIEVAL (Optional)                                          │
│  ──────────────────────────────────                                          │
│  Technique: Anthropic's Contextual Retrieval                                 │
│  Model: gpt-4o-mini (configurable)                                           │
│  Purpose: Add document-level context to each chunk                           │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ For each chunk, LLM generates context like:                          │    │
│  │ "This chunk discusses the relationship between exponentials and      │    │
│  │  the Laplace transform, following the introduction of complex        │    │
│  │  frequency in the previous section."                                 │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐               │
│  │   Semantic   │ ───► │     LLM      │ ───► │   Chunks +   │               │
│  │    Chunks    │      │ (gpt-4o-mini)│      │   Context    │               │
│  └──────────────┘      └──────────────┘      └──────────────┘               │
│                                                                              │
│  Cost: 1 API call per chunk (~1.8s each)                                    │
│  Toggle: DISABLE_CONTEXTUAL_RETRIEVAL=true                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  5. EMBEDDING GENERATION (Provider Locked)                                   │
│  ───────────────────────────────────────────                                 │
│  Default: voyage-3-large (Voyage AI) - 1024 dimensions                      │
│  Alternative: openai, bge, ollama (explicit config, NO FALLBACK)            │
│                                                                              │
│  IMPORTANT: Provider is locked on first ingestion. All chunks must use     │
│  the same provider. Switching requires: kb db migrate-embeddings --to X     │
│                                                                              │
│  Input: If context exists: "[context]\n\n[content]"                         │
│         Otherwise: "[content]" only                                          │
│                                                                              │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐               │
│  │   Chunks +   │ ───► │voyage-3-large│ ───► │   Vectors    │               │
│  │   Context    │      │  (1024-dim)  │      │  (1024-dim)  │               │
│  └──────────────┘      └──────────────┘      └──────────────┘               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  6. STORAGE                                                                  │
│  ──────────                                                                  │
│  Database: LanceDB (embedded vector database)                                │
│  Vector Index: IVF-PQ for approximate nearest neighbor                       │
│  FTS Index: Tantivy (BM25) for full-text search                             │
│                                                                              │
│  ┌──────────────┐      ┌──────────────┐                                     │
│  │   Vectors    │ ───► │   LanceDB    │                                     │
│  │  + Metadata  │      │  (persist)   │                                     │
│  └──────────────┘      └──────────────┘                                     │
│                                                                              │
│  Stored per chunk:                                                           │
│  - content (original text)                                                   │
│  - context (LLM-generated, if enabled)                                       │
│  - vector (1024 floats)                                                      │
│  - timestamp_start, timestamp_end                                            │
│  - source_id, tags, source_channel, source_type                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## RETRIEVAL PIPELINE (Question → Answer)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  1. QUERY TRANSFORMATION (HyDE)                                              │
│  ──────────────────────────────                                              │
│  Technique: Hypothetical Document Embeddings                                 │
│  Model: gpt-4o-mini                                                          │
│  Purpose: Bridge vocabulary gap between question and document                │
│                                                                              │
│  Problem: User asks "What is Laplace?" but document says                    │
│           "The transform converts time-domain to s-domain..."               │
│                                                                              │
│  Solution: Generate hypothetical answer, then embed that instead            │
│                                                                              │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐               │
│  │    Query     │ ───► │     LLM      │ ───► │ Hypothetical │               │
│  │  "What is    │      │   (HyDE)     │      │   Answer     │               │
│  │   Laplace?"  │      │              │      │              │               │
│  └──────────────┘      └──────────────┘      └──────────────┘               │
│                                                                              │
│  Output: "The Laplace transform is a mathematical operation that            │
│           converts a function of time into a function of complex            │
│           frequency, enabling analysis of differential equations..."        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  2. QUERY EMBEDDING                                                          │
│  ──────────────────                                                          │
│  Model: voyage-3-large (same as ingestion for consistency)                  │
│  Input: HyDE-transformed query (or original if HyDE disabled)               │
│                                                                              │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐               │
│  │ Hypothetical │ ───► │voyage-3-large│ ───► │    Query     │               │
│  │   Answer     │      │  (1024-dim)  │      │   Vector     │               │
│  └──────────────┘      └──────────────┘      └──────────────┘               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  3. HYBRID RETRIEVAL (Stage 1: High Recall) - SPLIT PIPELINE                 │
│  ───────────────────────────────────────────────────────────                 │
│  Technique: Vector Search + Full-Text Search + RRF Fusion                    │
│  Database: LanceDB                                                           │
│  Limit: 5x final limit (candidate pool for reranking)                       │
│                                                                              │
│  CRITICAL: FTS uses ORIGINAL query (not HyDE). If user searches "Error 404",│
│  BM25 must match literal "Error 404", not HyDE's "page not found issues".   │
│                                                                              │
│         ┌──────────────┐                                                     │
│         │    Query     │───────────────────────────────┐                     │
│         └──────┬───────┘                               │                     │
│                │                                       │ (Original Query)    │
│                ▼                                       ▼                     │
│         ┌──────────────┐                        ┌─────────────┐             │
│         │     HyDE     │                        │  FTS Search │             │
│         │  Generation  │                        │   (BM25)    │             │
│         └──────┬───────┘                        └──────┬──────┘             │
│                │ (Hypothetical Doc)                    │                     │
│                ▼                                       │                     │
│         ┌──────────────┐                               │                     │
│         │ Vector Embed │                               │                     │
│         │ (Voyage AI)  │                               │                     │
│         └──────┬───────┘                               │                     │
│                │                                       │                     │
│                ▼                                       │                     │
│         ┌──────────────┐                               │                     │
│         │   Vector     │                               │                     │
│         │   Search     │                               │                     │
│         └──────┬───────┘                               │                     │
│                │                                       │                     │
│                └───────────────┬───────────────────────┘                     │
│                                ▼                                             │
│                       ┌───────────────┐                                      │
│                       │  RRF Fusion   │  Reciprocal Rank Fusion (K=60)      │
│                       │               │  Combines rankings from both         │
│                       └───────────────┘                                      │
│                                │                                             │
│                                ▼                                             │
│                       ┌───────────────┐                                      │
│                       │  Candidates   │  ~50 chunks (high recall)           │
│                       └───────────────┘                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  4. CROSS-ENCODER RERANKING (Stage 2: High Precision)                        │
│  ────────────────────────────────────────────────────                        │
│  Model: ms-marco-MiniLM-L-12-v2 (FlashRank, ~34MB local)                    │
│  Technique: Cross-encoder scoring (query-document pairs)                     │
│                                                                              │
│  Why cross-encoder beats bi-encoder:                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ Bi-encoder (embedding):  query → vec_q   doc → vec_d   cos(q,d)     │    │
│  │   Fast (separate encoding), but loses query-doc interaction         │    │
│  │                                                                      │    │
│  │ Cross-encoder:  [query] [SEP] [doc] → transformer → score           │    │
│  │   Slow (joint encoding), but sees full query-doc interaction        │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  Input: Original query + "[context]\n\n[content]" for each candidate        │
│         Context field improves reranking when available!                     │
│                                                                              │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐               │
│  │  Candidates  │ ───► │Cross-Encoder │ ───► │   Reranked   │               │
│  │  (~50)       │      │  (pairwise)  │      │   Results    │               │
│  └──────────────┘      └──────────────┘      └──────────────┘               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  5. DEDUPLICATION (Stage 3: Diversity)                                       │
│  ─────────────────────────────────────                                       │
│  Technique: Jaccard similarity on word sets                                  │
│  Threshold: 0.9 (90% word overlap = duplicate)                              │
│                                                                              │
│  Why needed: Reranker optimizes for RELEVANCE, not DIVERSITY                │
│  Problem: 5 near-identical chunks about Laplace all score 0.99              │
│  Solution: Keep first, skip duplicates                                       │
│                                                                              │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐               │
│  │   Reranked   │ ───► │   Jaccard    │ ───► │   Diverse    │               │
│  │   Results    │      │   Dedup      │      │   Results    │               │
│  └──────────────┘      └──────────────┘      └──────────────┘               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  6. ENRICHMENT                                                               │
│  ─────────────                                                               │
│  Purpose: Add display metadata from Source table                             │
│  Added: source_title, source_url, timestamp_link                            │
│                                                                              │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐               │
│  │   Diverse    │ ───► │   Source     │ ───► │   Final      │               │
│  │   Results    │      │   Lookup     │      │   Results    │               │
│  └──────────────┘      └──────────────┘      └──────────────┘               │
│                                                                              │
│  Output per result:                                                          │
│  - chunk.content (text to display)                                          │
│  - chunk.context (if available)                                             │
│  - final_score (cross-encoder score, 0-1)                                   │
│  - source_title ("Why Laplace transforms...")                               │
│  - timestamp_link ("https://youtube.com/watch?v=xxx&t=246s")                │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Technique Summary

| Stage | Technique |
|-------|-----------|
| Download | yt-dlp (subtitle extraction) |
| Parsing | webvtt-py (timestamp preservation) |
| Chunking | **SEMANTIC**: all-MiniLM-L6-v2 (local, FREE) for topic detection |
| | *Fallback*: Sentence-boundary aware (~500 chars) |
| Context Gen | Contextual Retrieval (Anthropic technique, gpt-4o-mini) |
| Final Embedding | **voyage-3-large** (1024-dim) - provider locked on first ingestion |
| Storage | LanceDB (embedded vector DB + Tantivy FTS) |
| Query Transform | HyDE - Hypothetical Document Embeddings (gpt-4o-mini) |
| Retrieval | **SPLIT PIPELINE**: HyDE→Vector + Original→FTS + RRF K=60 |
| Reranking | Cross-Encoder (ms-marco-MiniLM-L-12-v2, local) |
| Deduplication | Jaccard similarity (0.9 threshold) |

---

## Configuration

### Embedding Provider (LOCKED - No Fallback)

**Critical**: The database locks to a single embedding provider on first ingestion.
Mixing embeddings from different providers corrupts search quality silently.

**Available Providers:**
| Provider | Model | API Key Required | Notes |
|----------|-------|------------------|-------|
| `voyage` | voyage-3-large | `VOYAGE_API_KEY` | Recommended, best quality |
| `openai` | text-embedding-3-large | `OPENAI_API_KEY` | Good alternative |
| `bge` | BAAI/bge-m3 | None (local) | Requires sentence-transformers |
| `ollama` | mxbai-embed-large | None (local) | Requires Ollama server |

**Provider Lock Behavior:**

| Scenario | Behavior |
|----------|----------|
| Fresh database | Locks to configured provider on first ingestion |
| Legacy database (has data, no metadata) | Trust On First Use - locks to current config, warns |
| Provider mismatch | **Fails immediately** with clear error |

**Switching Providers:**
```bash
# Check current configuration
kb config

# Migrate all chunks to a new provider
kb db migrate-embeddings --to openai

# Or skip confirmation
kb db migrate-embeddings --to voyage --yes
```

### Environment Variables
| Variable | Description |
|----------|-------------|
| `EMBEDDING_PROVIDER` | `voyage`, `openai`, `bge`, or `ollama` |
| `VOYAGE_API_KEY` | Voyage AI API key |
| `OPENAI_API_KEY` | OpenAI API key (also used for HyDE/Context) |
| `DISABLE_CONTEXTUAL_RETRIEVAL=true` | Skip context generation (faster) |
| `YOUTUBE_KB_DATA_DIR` | Custom data directory |

---

## Performance Characteristics

### Ingestion Time (per video)
| Mode | Time | API Calls |
|------|------|-----------|
| Without contextual retrieval | ~10s | 1 (embedding batch) |
| With contextual retrieval | ~2min | N+1 (1 per chunk + summary) |

### Search Latency
| Component | Time |
|-----------|------|
| HyDE transformation | ~500ms |
| Query embedding | ~100ms |
| Hybrid retrieval | ~50ms |
| Cross-encoder reranking | ~200ms |
| **Total** | **~2-3s** |
