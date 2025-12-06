# PARKED: Processing Time UX Issue

## Problem
Contextual retrieval makes ~278 API calls per video (5-10+ min), exceeds MCP timeout expectations.

## Current Workaround
`DISABLE_CONTEXTUAL_RETRIEVAL=true` env var in Claude Desktop config

## Options to Evaluate (When We Return to This)

### A) Always disable contextual retrieval in MCP
- **Pros**: Simple, fast processing
- **Cons**: Loses retrieval quality benefits

### B) Background processing with polling
- Return immediately: "Video queued for processing"
- User checks back later with `explore_library(view="source", source_id="...")`
- **Pros**: Non-blocking UX
- **Cons**: Requires job tracking, more complex

### C) MCP progress notifications
- Use MCP's streaming/progress protocol
- Send updates: "Processing chunk 50/278..."
- **Pros**: Real-time feedback
- **Cons**: Not all MCP clients support this well

### D) Hybrid: Fast ingest, enrich later
- Quick ingest without context (30 seconds)
- Background enrichment adds context over time
- **Pros**: Best of both worlds
- **Cons**: Most complex to implement

## Why Parked
Need clean data models as foundation before tackling async architecture.
Will revisit after Issue 2 (Data Optimization) is complete.
