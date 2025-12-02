"""
LanceDB database management.

Note on timestamps: PyArrow timestamp("us") expects datetime objects, not ISO strings.
The Pydantic models use datetime which is correct - avoid accidental string conversion when inserting.

Note on embeddings: Embedding initialization is lazy. This Database class does NOT
initialize embeddings. Embeddings are created in the repository/service layer when
actually adding chunks.
"""
import lancedb
from pathlib import Path
from typing import Optional
import pyarrow as pa
import logging

from .config import settings

logger = logging.getLogger(__name__)

# Define PyArrow schemas for LanceDB tables
SOURCES_SCHEMA = pa.schema([
    pa.field("id", pa.string()),
    pa.field("source_type", pa.string()),
    pa.field("source_id", pa.string()),
    pa.field("title", pa.string()),
    pa.field("channel", pa.string()),
    pa.field("description", pa.string()),
    pa.field("url", pa.string()),
    pa.field("metadata", pa.string()),  # JSON string
    pa.field("tags", pa.list_(pa.string())),
    pa.field("collections", pa.list_(pa.string())),
    pa.field("user_summary", pa.string()),
    pa.field("is_processed", pa.bool_()),
    pa.field("chunk_count", pa.int32()),
    pa.field("embedding_model", pa.string()),
    pa.field("created_at", pa.timestamp("us")),
    pa.field("updated_at", pa.timestamp("us")),
])


def get_chunks_schema(vector_dim: int = 1024) -> pa.Schema:
    """Generate chunks schema with specified vector dimensions."""
    return pa.schema([
        pa.field("id", pa.string()),
        pa.field("source_id", pa.string()),
        pa.field("content", pa.string()),
        pa.field("chunk_index", pa.int32()),
        pa.field("vector", pa.list_(pa.float32(), vector_dim)),
        pa.field("timestamp_start", pa.float64()),
        pa.field("timestamp_end", pa.float64()),
        pa.field("source_type", pa.string()),
        pa.field("source_title", pa.string()),
        pa.field("source_channel", pa.string()),
        pa.field("tags", pa.list_(pa.string())),
        pa.field("collections", pa.list_(pa.string())),
        pa.field("embedding_model", pa.string()),
        # Contextual retrieval fields
        pa.field("context", pa.string()),
        pa.field("context_model", pa.string()),
        # Future SOTA RAG fields
        pa.field("parent_id", pa.string()),
        pa.field("speakers", pa.list_(pa.string())),
        pa.field("chapter_index", pa.int32()),
        pa.field("created_at", pa.timestamp("us")),
    ])


class Database:
    """LanceDB database wrapper."""

    _instance: Optional["Database"] = None

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or settings.db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(str(self.db_path))
        self._ensure_tables()
        self._migrate_schemas()

    @classmethod
    def get_instance(cls) -> "Database":
        """Get or create singleton database instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _ensure_tables(self):
        """Ensure required tables exist with FTS index for hybrid search."""
        existing_tables = self._db.table_names()

        if "sources" not in existing_tables:
            # Create empty sources table with schema
            self._db.create_table("sources", schema=SOURCES_SCHEMA)

        if "chunks" not in existing_tables:
            # Create empty chunks table with schema
            chunks_schema = get_chunks_schema(settings.embedding.dimensions)
            self._db.create_table("chunks", schema=chunks_schema)

        # Create FTS index on chunks.content for hybrid search
        if "chunks" in self._db.table_names():
            chunks_table = self._db.open_table("chunks")
            try:
                chunks_table.create_fts_index("content", replace=True)
            except Exception:
                # Index may already exist or table may be empty
                pass

    def _migrate_schemas(self):
        """
        Add missing columns to existing tables for forward compatibility.

        This handles schema evolution when new fields are added to the models.
        LanceDB supports adding nullable columns to existing tables.
        """
        if "chunks" not in self._db.table_names():
            return

        try:
            chunks_table = self._db.open_table("chunks")
            existing_names = {f.name for f in chunks_table.schema}
            expected_schema = get_chunks_schema(settings.embedding.dimensions)

            # Find missing fields
            missing_fields = []
            for field in expected_schema:
                if field.name not in existing_names:
                    missing_fields.append(field.name)

            if missing_fields:
                logger.info(f"Schema migration needed: adding {missing_fields} to chunks table")
                # LanceDB requires recreating table or using add_columns
                # For now, we'll add columns with null values using pyarrow
                # Note: LanceDB's add_columns API may vary by version
                try:
                    # Try the newer API first
                    for field_name in missing_fields:
                        # Get the field definition from expected schema
                        for field in expected_schema:
                            if field.name == field_name:
                                # Add column with null default
                                # LanceDB will fill existing rows with null
                                chunks_table.add_columns({field_name: None})
                                logger.info(f"Added column '{field_name}' to chunks table")
                                break
                except Exception as e:
                    # If add_columns fails, log warning but continue
                    # The repository layer will handle missing fields gracefully
                    logger.warning(
                        f"Could not auto-migrate schema: {e}. "
                        f"New chunks will have all fields, existing chunks may lack: {missing_fields}"
                    )
        except Exception as e:
            logger.warning(f"Schema migration check failed: {e}")

    @property
    def sources(self):
        """Get sources table."""
        return self._db.open_table("sources")

    @property
    def chunks(self):
        """Get chunks table."""
        return self._db.open_table("chunks")

    def reset(self):
        """Drop and recreate all tables. USE WITH CAUTION."""
        for table_name in self._db.table_names():
            self._db.drop_table(table_name)
        self._ensure_tables()
        # Reset singleton so next get_instance creates fresh connection
        Database._instance = None


def get_db() -> Database:
    """Get database instance."""
    return Database.get_instance()
