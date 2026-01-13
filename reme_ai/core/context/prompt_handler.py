"""Module for managing and formatting prompt templates from files or dictionaries."""

from pathlib import Path

import yaml
from loguru import logger

from .base_context import BaseContext
from .service_context import C


class PromptHandler(BaseContext):
    """A context-aware handler for loading, retrieving, and formatting prompt templates."""

    def __init__(self, language: str = "", **kwargs):
        """Initialize the handler with a specific language and optional context data."""
        super().__init__(**kwargs)
        self.language: str = language or C.language

    def load_prompt_by_file(self, prompt_file_path: Path | str = None):
        """Load prompt configurations from a YAML file into the context."""
        if prompt_file_path is None:
            return self

        if isinstance(prompt_file_path, str):
            prompt_file_path = Path(prompt_file_path)

        if not prompt_file_path.exists():
            return self

        with prompt_file_path.open(encoding="utf-8") as f:
            # Load YAML content using the full loader
            prompt_dict = yaml.load(f, yaml.FullLoader)
            self.load_prompt_dict(prompt_dict)
        return self

    def load_prompt_dict(self, prompt_dict: dict = None):
        """Merge a dictionary of prompt strings into the current context."""
        if not prompt_dict:
            return self

        for key, value in prompt_dict.items():
            if isinstance(value, str):
                if key in self:
                    logger.warning(f"Overwriting prompt key={key}, old_value={self[key]}, new_value={value}")
                else:
                    logger.debug(f"Adding new prompt key={key}, value={value}")
                self[key] = value
        return self

    def get_prompt(self, prompt_name: str):
        """Retrieve a prompt by name, automatically appending the language suffix if needed."""
        key: str = prompt_name
        if self.language and not key.endswith(self.language.strip()):
            key += "_" + self.language.strip()

        assert key in self, f"prompt_name={key} not found."
        return self[key]

    def prompt_format(self, prompt_name: str, **kwargs) -> str:
        """Format a prompt by filtering flagged lines and filling template variables."""
        prompt = self.get_prompt(prompt_name)

        # Separate boolean flags from string formatting arguments
        flag_kwargs = {k: v for k, v in kwargs.items() if isinstance(v, bool)}
        other_kwargs = {k: v for k, v in kwargs.items() if not isinstance(v, bool)}

        if flag_kwargs:
            split_prompt = []
            for line in prompt.strip().split("\n"):
                hit = False
                hit_flag = True
                for key, flag in flag_kwargs.items():
                    if not line.startswith(f"[{key}]"):
                        continue

                    hit = True
                    hit_flag = flag
                    # Remove the flag prefix from the line
                    line = line.strip(f"[{key}]")
                    break

                # Include line if no flag is present or if the flag evaluates to True
                if not hit:
                    split_prompt.append(line)
                elif hit_flag:
                    split_prompt.append(line)

            prompt = "\n".join(split_prompt)

        if other_kwargs:
            # Apply standard Python string formatting
            prompt = prompt.format(**other_kwargs)

        return prompt
