"""Add meta memory operation for adding memory metadata."""

import json

from loguru import logger

from ..base_memory_tool import BaseMemoryTool
from ...core.context import C
from ...core.enumeration import MemoryType


@C.register_op()
class AddMetaMemory(BaseMemoryTool):
    """Add memory metadata (memory_type and memory_target) to meta storage.

    Supports single/multiple addition modes via `enable_multiple` parameter.
    """

    def _build_item_schema(self) -> tuple[dict, list[str]]:
        """Build shared schema properties and required fields for meta memory items.

        Returns:
            Tuple of (properties dict, required fields list).
        """
        properties = {
            "memory_type": {
                "type": "string",
                "description": self.get_prompt("memory_type"),
                "enum": [MemoryType.PERSONAL.value, MemoryType.PROCEDURAL.value],
            },
            "memory_target": {
                "type": "string",
                "description": self.get_prompt("memory_target"),
            },
        }
        required = ["memory_type", "memory_target"]
        return properties, required

    def _build_parameters(self) -> dict:
        """Build input schema for single meta memory addition."""
        properties, required = self._build_item_schema()
        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    def _build_multiple_parameters(self) -> dict:
        """Build input schema for multiple meta memory addition."""
        item_properties, required_fields = self._build_item_schema()
        return {
            "type": "object",
            "properties": {
                "meta_memories": {
                    "type": "array",
                    "description": self.get_prompt("meta_memories"),
                    "items": {
                        "type": "object",
                        "properties": item_properties,
                        "required": required_fields,
                    },
                },
            },
            "required": ["meta_memories"],
        }

    def _load_meta_memories(self) -> list[dict]:
        """Load existing meta memories from cache."""
        return self.meta_memory.load("meta_memories") or []

    def _save_meta_memories(self, memories: list[dict]) -> bool:
        """Save meta memories to cache."""
        return self.meta_memory.save("meta_memories", memories)

    @staticmethod
    def _filter_memory_type_target(memory_type: str, memory_target: str, existing_set: set) -> bool:
        result = (
            memory_type in [MemoryType.PERSONAL.value, MemoryType.PROCEDURAL.value]
            and memory_target
            and (memory_type, memory_target) not in existing_set
        )
        if result:
            existing_set.add((memory_type, memory_target))
        return result

    async def execute(self):
        """Execute addition: load existing, merge with new, and save.

        Duplicates (same memory_type and memory_target) are skipped.
        """
        existing_memories: list[dict] = self._load_meta_memories()
        existing_set = {(m["memory_type"], m["memory_target"]) for m in existing_memories}

        # Build new memories to add based on mode
        new_memories: list[dict] = []
        if self.enable_multiple:
            meta_memories: list[dict] = self.context.get("meta_memories", [])
            for mem in meta_memories:
                memory_type = mem.get("memory_type", "")
                memory_target = mem.get("memory_target", "")
                if self._filter_memory_type_target(memory_type, memory_target, existing_set):
                    new_memories.append({"memory_type": memory_type, "memory_target": memory_target})

        else:
            memory_type = self.context.get("memory_type", "")
            memory_target = self.context.get("memory_target", "")
            if self._filter_memory_type_target(memory_type, memory_target, existing_set):
                new_memories.append({"memory_type": memory_type, "memory_target": memory_target})

        if not new_memories:
            self.output = "No new meta memories to add (all entries already exist or invalid)."
            return

        # Merge and save
        all_memories = existing_memories + new_memories
        self._save_meta_memories(all_memories)

        # Format output
        added_str = json.dumps(new_memories, ensure_ascii=False)
        self.output = f"Successfully added {len(new_memories)} meta memory entries: {added_str}"
        logger.info(self.output)
