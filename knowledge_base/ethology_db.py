"""
AnimalVox AI — Ethology Knowledge Base (Stage 4)
ChromaDB vector database with 500+ peer-reviewed ethological entries.
Provides RAG context for LLM translation layer.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# Try ChromaDB, fall back to in-memory search
try:
    import chromadb
    from chromadb.config import Settings
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False
    logger.warning("ChromaDB not installed — using in-memory fallback")


class EthologyKnowledgeBase:
    """
    Scientific knowledge base for animal vocalization interpretation.
    Each entry maps (species, behavior) → behavioral meaning + translation templates.
    """

    def __init__(self, db_path: str = None, use_chroma: bool = True):
        self.entries: List[Dict] = []

        if use_chroma and HAS_CHROMADB:
            db_path = db_path or str(Path(__file__).parent / "chroma_db")
            self.client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=db_path,
                anonymized_telemetry=False
            ))
            self.collection = self.client.get_or_create_collection(
                name="ethology_kb",
                metadata={"description": "Animal behavior ethology entries"}
            )
            self.use_chroma = True
            logger.info(f"ChromaDB initialized at {db_path}")
        else:
            self.use_chroma = False
            logger.info("Using in-memory knowledge base")

    def add_entry(self, entry: Dict):
        """Add a single ethology entry."""
        self.entries.append(entry)

        if self.use_chroma:
            doc_text = (
                f"{entry['common_name']} ({entry['species_id']}). "
                f"Behavior: {entry['behavior_class']}. "
                f"Call pattern: {entry.get('call_pattern', 'N/A')}. "
                f"Meaning: {entry.get('behavioral_meaning', 'N/A')}. "
                f"Response: {entry.get('behavioral_response', 'N/A')}."
            )
            self.collection.add(
                documents=[doc_text],
                metadatas=[{
                    "species_id": entry["species_id"],
                    "species_group": entry["species_group"],
                    "behavior_class": entry["behavior_class"],
                    "confidence": entry.get("confidence", "medium")
                }],
                ids=[f"{entry['species_id']}_{entry['behavior_class']}_{len(self.entries)}"]
            )

    def add_entries_bulk(self, entries: List[Dict]):
        """Add multiple entries at once."""
        for entry in entries:
            self.add_entry(entry)
        logger.info(f"Added {len(entries)} entries to knowledge base (total: {len(self.entries)})")

    def query(self, species_group: str, behavior_class: str, n_results: int = 3) -> List[Dict]:
        """Query knowledge base for relevant ethology context."""
        if self.use_chroma:
            query_text = f"{species_group} {behavior_class} vocalization behavior"
            results = self.collection.query(
                query_texts=[query_text], n_results=n_results,
                where={"species_group": species_group}
            )
            matched = []
            for i, doc in enumerate(results.get("documents", [[]])[0]):
                meta = results["metadatas"][0][i] if results.get("metadatas") else {}
                # Find full entry
                for e in self.entries:
                    if (e.get("species_group") == meta.get("species_group") and
                            e.get("behavior_class") == meta.get("behavior_class")):
                        matched.append(e)
                        break
            return matched[:n_results]
        else:
            # In-memory fallback
            return [e for e in self.entries
                    if e.get("species_group") == species_group
                    and e.get("behavior_class") == behavior_class][:n_results]

    def get_translation_template(self, species_group: str, behavior_class: str, intensity: float) -> str:
        """Get intensity-appropriate translation template."""
        entries = self.query(species_group, behavior_class, n_results=1)
        if not entries:
            return ""

        entry = entries[0]
        modifiers = entry.get("intensity_modifiers", {})
        if intensity >= 0.7:
            return modifiers.get("high", entry.get("translation_template", ""))
        elif intensity >= 0.4:
            return modifiers.get("medium", entry.get("translation_template", ""))
        return modifiers.get("low", entry.get("translation_template", ""))

    def get_context_for_llm(self, species_group: str, behavior_classes: List[str]) -> str:
        """Build rich context string for LLM prompt injection."""
        contexts = []
        for bc in behavior_classes:
            entries = self.query(species_group, bc, n_results=2)
            for e in entries:
                ctx = (
                    f"[{e['common_name']}] {e['behavior_class']}: "
                    f"{e.get('behavioral_meaning', 'Unknown')}. "
                    f"Pattern: {e.get('call_pattern', 'N/A')}. "
                    f"Source: {e.get('source', 'ethology literature')}"
                )
                contexts.append(ctx)
        return " | ".join(contexts) if contexts else "No specific ethology reference available."

    def load_seed_data(self, seed_dir: str = None):
        """Load all JSON seed data files from seed_data directory."""
        if seed_dir is None:
            seed_dir = str(Path(__file__).parent / "seed_data")

        seed_path = Path(seed_dir)
        if not seed_path.exists():
            logger.warning(f"Seed directory not found: {seed_dir}")
            return

        count = 0
        for json_file in sorted(seed_path.glob("*.json")):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                entries = data if isinstance(data, list) else [data]
                self.add_entries_bulk(entries)
                count += len(entries)
            except Exception as e:
                logger.error(f"Failed to load {json_file}: {e}")

        logger.info(f"Loaded {count} seed entries from {seed_dir}")

    @property
    def size(self) -> int:
        return len(self.entries)
