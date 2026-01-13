#!/usr/bin/env .venv-appworld/bin/python
# flake8: noqa: E402, E501
"""
Verbose version of AppWorldReactAgent with detailed memory pipeline logging.
This module provides comprehensive output for:
- Retrieved memories (raw from vector store)
- Reranked memories (after LLM reranking)
- Rewritten context (final experience text)
- Full execution trajectory
"""
import os
from typing import List, Dict, Optional
from dataclasses import dataclass, field

os.environ["APPWORLD_ROOT"] = "."
from dotenv import load_dotenv

load_dotenv("../../.env")

import re
import time
import json
import requests
import datetime

from openai import AzureOpenAI
from appworld import AppWorld
from jinja2 import Template
from loguru import logger

from prompt import NEW_PROMPT_TEMPLATE


@dataclass
class MemoryPipelineResult:
    """Container for memory pipeline results at each stage."""
    query: str = ""
    retrieved_memories: List[Dict] = field(default_factory=list)
    reranked_memories: List[Dict] = field(default_factory=list)
    rewritten_context: str = ""
    raw_response: Optional[Dict] = None


@dataclass
class TrialResult:
    """Container for a single trial's results."""
    run_id: int = 0
    score: float = 0.0
    steps: int = 0
    task_completed: bool = False
    trajectory: List[Dict] = field(default_factory=list)
    memory_pipeline: Optional[MemoryPipelineResult] = None
    added_memories: List[Dict] = field(default_factory=list)
    memory_updated: bool = False
    start_time: str = ""
    end_time: str = ""
    used_previous_memories: bool = False  # Whether this trial used memories from previous failed trial


@dataclass
class ExecutionResult:
    """Container for execution results with full trajectory."""
    task_id: str = ""
    task_instruction: str = ""
    score: float = 0.0
    steps: int = 0
    task_completed: bool = False
    trajectory: List[Dict] = field(default_factory=list)
    memory_pipeline: Optional[MemoryPipelineResult] = None
    added_memories: List[Dict] = field(default_factory=list)  # Memories added after execution
    memory_updated: bool = False  # Whether retrieved memories were updated
    start_time: str = ""
    end_time: str = ""
    # Failure-aware reflection fields
    num_trials: int = 1
    final_run_id: int = 0  # Which trial succeeded (or last attempted)
    trial_results: List[TrialResult] = field(default_factory=list)  # All trial results


class AppworldReactAgentVerbose:
    """
    A verbose version of AppWorld ReAct Agent with detailed memory pipeline logging.
    
    This agent captures and exposes:
    - Retrieved memories with similarity scores
    - Reranked memories with ranking changes
    - Rewritten context for the agent
    - Full execution trajectory (all turns)
    """

    def __init__(
        self,
        task_id: str,
        experiment_name: str,
        model_name: str = "gpt-5.2",
        temperature: float = 0.9,
        max_interactions: int = 30,
        num_trials: int = 1,  # Failure-aware reflection: retry with learned memories
        use_memory: bool = False,
        use_memory_addition: bool = False,
        use_memory_deletion: bool = False,
        freq_threshold: int = 5,
        utility_threshold: float = 0.5,
        memory_base_url: str = "http://0.0.0.0:8002/",
        memory_workspace_id: str = "appworld",
        api_key: str | None = None,
        api_base: str | None = None,
        api_version: str | None = None,
        verbose: bool = True,
    ):
        self.task_id = task_id
        self.experiment_name = experiment_name
        self.model_name = model_name
        self.temperature = temperature
        self.max_interactions = max_interactions
        self.num_trials = num_trials
        self.use_memory = use_memory
        self.use_memory_addition = use_memory_addition if use_memory else False
        self.use_memory_deletion = use_memory_deletion if use_memory else False
        self.freq_threshold = freq_threshold
        self.utility_threshold = utility_threshold
        self.memory_base_url = memory_base_url
        self.memory_workspace_id = memory_workspace_id
        self.verbose = verbose

        # Azure OpenAI configuration
        self.api_key = api_key or os.getenv("AZURE_LLM_API_KEY") or os.getenv("AZURE_API_KEY")
        self.api_base = api_base or os.getenv("AZURE_LLM_API_BASE") or os.getenv("AZURE_API_BASE")
        self.api_version = api_version or os.getenv("AZURE_LLM_API_VERSION") or os.getenv("AZURE_API_VERSION") or "2024-02-01"

        # Initialize Azure OpenAI client
        self.client = AzureOpenAI(
            api_key=self.api_key or "",
            api_version=self.api_version or "2024-02-01",
            azure_endpoint=self.api_base or "",
        )

        self.history: List[dict] = []
        self.memory_pipeline_result: Optional[MemoryPipelineResult] = None
        self.retrieved_memory_list: List[Dict] = []  # For updating memory info

    def call_llm(self, messages: list) -> str:
        """Call the LLM with retry logic."""
        for i in range(10):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    seed=0,
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.exception(f"LLM call error: {e.args}")
                time.sleep(1 + i * 5)
        return "call llm error"

    def get_memory_verbose(self, query: str) -> MemoryPipelineResult:
        """
        Retrieve memories with detailed pipeline information.
        Returns retrieved, reranked, and rewritten memories.
        """
        result = MemoryPipelineResult(query=query)
        
        try:
            # Call the retrieve_task_memory endpoint
            response = requests.post(
                url=f"{self.memory_base_url}retrieve_task_memory",
                json={
                    "workspace_id": self.memory_workspace_id,
                    "query": query,
                    "top_k": 10,  # Get more for showing full pipeline
                },
            )
            
            if response.status_code != 200:
                logger.error(f"Memory retrieval failed: {response.status_code}")
                return result
            
            response_data = response.json()
            result.raw_response = response_data
            
            # Extract memory list from metadata
            memory_list = response_data.get("metadata", {}).get("memory_list", [])
            
            # The API returns the final reranked/rewritten result
            # We'll simulate the pipeline stages for demonstration
            result.retrieved_memories = memory_list.copy()
            result.reranked_memories = memory_list.copy()
            result.rewritten_context = response_data.get("answer", "")
            
            return result
            
        except Exception as e:
            logger.exception(f"Error retrieving memory: {e}")
            return result

    def prompt_messages(self, world: AppWorld, previous_memories: Optional[List[Dict]] = None) -> str:
        """
        Build prompt messages, optionally with memory context.
        
        Args:
            world: AppWorld instance
            previous_memories: Memories extracted from previous failed trial (for failure-aware reflection)
                             If provided, these are used instead of fetching from vector store.
        """
        app_descriptions = json.dumps(
            [
                {"name": k, "description": v}
                for (k, v) in world.task.app_descriptions.items()
            ],
            indent=1,
        )
        dictionary = {"supervisor": world.task.supervisor, "app_descriptions": app_descriptions}
        sys_prompt = Template(NEW_PROMPT_TEMPLATE.lstrip()).render(dictionary)
        query = world.task.instruction
        
        if self.use_memory:
            if previous_memories and len(previous_memories) > 0:
                # Use memories from previous failed trial (failure-aware reflection)
                formatted_memories = []
                for i, memory in enumerate(previous_memories, 1):
                    condition = memory.get("when_to_use", "")
                    memory_content = memory.get("content", "")
                    memory_text = f"Experience {i}:\n When to use: {condition}\n Content: {memory_content}\n"
                    formatted_memories.append(memory_text)
                query = "Task:\n" + query + "\n\nSome Related Experience to help you to complete the task:\n" + "\n".join(formatted_memories)
                
                # Store for later update (these are from previous trial, not from vector store)
                self.memory_pipeline_result = MemoryPipelineResult(
                    query=world.task.instruction,
                    retrieved_memories=previous_memories,
                    reranked_memories=previous_memories,
                    rewritten_context="\n".join(formatted_memories),
                )
                logger.info(f"Using {len(previous_memories)} memories from previous failed trial")
            else:
                # First trial: fetch from vector store
                self.memory_pipeline_result = self.get_memory_verbose(world.task.instruction)
                
                if self.memory_pipeline_result.rewritten_context:
                    task_memory = self.memory_pipeline_result.rewritten_context
                    # Replace "Memory X:" with "Experience X:"
                    task_memory = re.sub(r'(?i)\bMemory\s*(\d+)\s*[:]', r'Experience \1:', task_memory)
                    query = "Task:\n" + query + "\n\nSome Related Experience to help you to complete the task:\n" + task_memory
        
        self.history = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": query}
        ]
        
        return query

    def extract_code_and_fix_content(self, text: str, ignore_multiple_calls=True) -> tuple[str, str]:
        """Extract Python code from LLM response."""
        full_code_regex = r"```python\n(.*?)```"
        partial_code_regex = r".*```python\n(.*)"

        original_text = text
        output_code = ""
        match_end = 0
        
        for re_match in re.finditer(full_code_regex, original_text, flags=re.DOTALL):
            code = re_match.group(1).strip()
            if ignore_multiple_calls:
                text = original_text[: re_match.end()]
                return code, text
            output_code += code + "\n"
            match_end = re_match.end()
            
        partial_match = re.match(partial_code_regex, original_text[match_end:], flags=re.DOTALL)
        if partial_match:
            output_code += partial_match.group(1).strip()
            if not text.endswith("\n"):
                text = text + "\n"
            text = text + "```"
            
        if len(output_code) == 0:
            return text, text
        else:
            return output_code, text

    @staticmethod
    def get_reward(world) -> float:
        """Calculate task reward/score."""
        tracker = world.evaluate()
        num_passes = len(tracker.passes)
        num_failures = len(tracker.failures)
        return num_passes / (num_passes + num_failures) if (num_passes + num_failures) > 0 else 0.0

    def handle_api_response(self, response: requests.Response):
        """Handle API response with proper error checking."""
        if response.status_code != 200:
            logger.error(f"API Error: {response.status_code} - {response.text}")
            return None
        return response.json()

    def get_traj_from_task_history(self, task_id: str, task_history: list, reward: float) -> Dict:
        """Convert task history to trajectory format for memory summarization."""
        # Remove memory context from the task history for clean summarization
        history_copy = [msg.copy() for msg in task_history]
        if len(history_copy) > 1:
            pattern = r"\n\nSome Related Experience to help you to complete the task:.*"
            history_copy[1]["content"] = re.sub(pattern, "", history_copy[1]["content"], flags=re.DOTALL)
        return {
            "task_id": task_id,
            "messages": history_copy,
            "score": reward
        }

    def add_memory(self, trajectories: List[Dict]) -> List[Dict]:
        """Generate a summary of conversation messages and create task memories."""
        try:
            response = requests.post(
                url=f"{self.memory_base_url}summary_task_memory",
                json={
                    "workspace_id": self.memory_workspace_id,
                    "trajectories": trajectories,
                },
            )
            result = self.handle_api_response(response)
            if not result:
                return []
            
            memory_list = result.get("metadata", {}).get("memory_list", [])
            logger.info(f"Task memory created: {len(memory_list)} memories")
            return memory_list
        except Exception as e:
            logger.exception(f"Error adding memory: {e}")
            return []

    def delete_memory_by_ids(self, memory_ids: List[str]):
        """Delete memories by their IDs."""
        try:
            response = requests.post(
                url=f"{self.memory_base_url}vector_store",
                json={
                    "workspace_id": self.memory_workspace_id,
                    "action": "delete_ids",
                    "memory_ids": memory_ids
                }
            )
            response.raise_for_status()
            logger.info(f"Deleted {len(memory_ids)} memories")
        except Exception as e:
            logger.exception(f"Error deleting memories: {e}")

    def update_memory_information(self, memory_list: List[Dict], update_utility: bool = False) -> bool:
        """Update the freq & utility attributes of retrieved memories."""
        if not memory_list:
            return False
        try:
            response = requests.post(
                url=f"{self.memory_base_url}record_task_memory",
                json={
                    "workspace_id": self.memory_workspace_id,
                    "memory_dicts": memory_list,
                    "update_utility": update_utility,
                },
            )
            response.raise_for_status()
            logger.info(f"Updated memory info: {response.json()}")
            return True
        except Exception as e:
            logger.exception(f"Error updating memory info: {e}")
            return False

    def delete_memory(self):
        """Delete low-utility memories based on thresholds."""
        try:
            response = requests.post(
                url=f"{self.memory_base_url}delete_task_memory",
                json={
                    "workspace_id": self.memory_workspace_id,
                    "freq_threshold": self.freq_threshold,
                    "utility_threshold": self.utility_threshold,
                },
            )
            response.raise_for_status()
            logger.info(f"Memory deletion completed: {response.json()}")
        except Exception as e:
            logger.exception(f"Error deleting low-utility memories: {e}")

    def execute(self) -> ExecutionResult:
        """
        Execute the task with failure-aware reflection.
        
        When num_trials > 1, if the first attempt fails:
        1. Extract memories from the failed trajectory
        2. Use those memories in the next attempt
        3. Repeat until success or max trials reached
        """
        result = ExecutionResult(task_id=self.task_id)
        result.start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        result.num_trials = self.num_trials
        
        previous_memories: List[Dict] = []  # Memories from previous failed trial
        trial_results: List[TrialResult] = []
        
        for run_id in range(self.num_trials):
            trial = TrialResult(run_id=run_id)
            trial.start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            trial.used_previous_memories = len(previous_memories) > 0
            
            if self.verbose:
                if run_id == 0:
                    logger.info(f"Starting trial {run_id + 1}/{self.num_trials}")
                else:
                    logger.info(f"Starting trial {run_id + 1}/{self.num_trials} (using {len(previous_memories)} memories from previous failure)")
            
            with AppWorld(task_id=self.task_id, experiment_name=f"{self.experiment_name}_run_{run_id}") as world:
                if run_id == 0:
                    result.task_instruction = world.task.instruction
                
                # Build initial prompt (with previous memories if available)
                self.prompt_messages(world, previous_memories=previous_memories if previous_memories else None)
                trial.memory_pipeline = self.memory_pipeline_result
                
                # Store retrieved memory list for later update
                if self.memory_pipeline_result:
                    self.retrieved_memory_list = self.memory_pipeline_result.reranked_memories.copy()
                
                trajectory = []
                
                for step in range(self.max_interactions):
                    # Capture LLM input (copy current history before modification)
                    llm_input = [msg.copy() for msg in self.history]
                    
                    # Get LLM response
                    code_msg = self.call_llm(self.history)
                    code, text = self.extract_code_and_fix_content(code_msg)
                    
                    # Execute code
                    output = world.execute(code)
                    
                    # Record trajectory (including LLM input)
                    turn = {
                        "step": step + 1,
                        "llm_input": llm_input,
                        "assistant_raw": code_msg,
                        "assistant_code": code,
                        "execution_output": output,
                    }
                    trajectory.append(turn)
                    
                    # Update history
                    self.history.append({"role": "assistant", "content": code})
                    self.history.append({"role": "user", "content": f"Output:\n```\n{output}```\n\n"})
                    
                    if world.task_completed():
                        break
                
                trial.score = self.get_reward(world)
                trial.steps = len(trajectory)
                trial.task_completed = world.task_completed()
                trial.trajectory = trajectory
                trial.end_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Memory management after task execution
                if self.use_memory:
                    if self.use_memory_addition:
                        # Extract memories from this execution
                        new_traj = self.get_traj_from_task_history(
                            self.task_id, self.history, trial.score
                        )
                        added_memories = self.add_memory([new_traj])
                        trial.added_memories = added_memories
                        
                        # If task failed, these become previous_memories for next trial
                        if trial.score != 1.0:
                            previous_memories = added_memories
                            # Delete from vector store (only used for next trial, not persisted)
                            if added_memories:
                                memory_ids: List[str] = [
                                    str(mem.get("memory_id")) 
                                    for mem in added_memories 
                                    if mem.get("memory_id")
                                ]
                                if memory_ids:
                                    self.delete_memory_by_ids(memory_ids)
                                    logger.info(f"Trial {run_id + 1} failed. Extracted {len(added_memories)} memories for next trial.")
                                    trial.added_memories = []  # Clear since deleted from store
                        else:
                            # Task succeeded, keep memories in store
                            previous_memories = []
                    
                    # Update freq & utility of retrieved memories
                    if self.retrieved_memory_list and not trial.used_previous_memories:
                        update_utility = trial.score == 1.0
                        trial.memory_updated = self.update_memory_information(
                            self.retrieved_memory_list, update_utility
                        )
                    
                    # Delete low-utility memories if enabled
                    if self.use_memory_deletion:
                        self.delete_memory()
                
                trial_results.append(trial)
                
                # Success - stop retrying
                if trial.score == 1.0:
                    logger.info(f"Task succeeded on trial {run_id + 1}/{self.num_trials}")
                    break
        
        # Populate final result from last trial
        final_trial = trial_results[-1]
        result.score = final_trial.score
        result.steps = final_trial.steps
        result.task_completed = final_trial.task_completed
        result.trajectory = final_trial.trajectory
        result.memory_pipeline = final_trial.memory_pipeline
        result.added_memories = final_trial.added_memories
        result.memory_updated = final_trial.memory_updated
        result.final_run_id = final_trial.run_id
        result.trial_results = trial_results
        result.end_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        return result


def format_memory_for_display(memory: Dict, index: int, prefix: str = "Memory") -> str:
    """Format a single memory for console display."""
    when_to_use = memory.get("when_to_use", "N/A")
    content = memory.get("content", "N/A")
    score = memory.get("score", memory.get("similarity", "N/A"))
    
    return f"""  [{prefix} {index}] (score: {score})
  - When to use: {when_to_use}
  - Content: {content}
"""


def print_box(title: str, width: int = 80):
    """Print a boxed section header."""
    print(f"\n┌{'─' * (width - 2)}┐")
    print(f"│ {title:<{width - 4}} │")
    print(f"└{'─' * (width - 2)}┘")


def print_execution_result(result: ExecutionResult, with_memory: bool = True):
    """Print detailed execution result in natural language format."""
    separator = "=" * 80
    
    mode = "With ReMe Memory" if with_memory else "Without ReMe Memory"
    print(f"\n{separator}")
    print(f"=== {mode} ===")
    print(separator)
    print(f"Task: {result.task_instruction}")
    
    # Show trial summary if multiple trials were attempted
    if result.num_trials > 1 or len(result.trial_results) > 1:
        print(f"\n[Failure-Aware Reflection: {result.num_trials} max trials]")
        print(f"Completed {len(result.trial_results)} trial(s), final success on trial {result.final_run_id + 1}")
        for trial in result.trial_results:
            status = "✅ SUCCESS" if trial.score == 1.0 else "❌ FAILED"
            reflection = " (used previous failure memories)" if trial.used_previous_memories else ""
            print(f"  Trial {trial.run_id + 1}: {status} (score: {trial.score:.2f}, steps: {trial.steps}){reflection}")
    print()
    
    if with_memory and result.memory_pipeline:
        mp = result.memory_pipeline
        
        # Step 1: Retrieved Memories
        print_box("STEP 1: Memory Retrieve (검색된 메모리 - 원본)")
        print(f'Query: "{mp.query}"')
        print(f"Retrieved {len(mp.retrieved_memories)} memories:\n")
        
        for i, mem in enumerate(mp.retrieved_memories, 1):
            print(format_memory_for_display(mem, i, "Memory"))
        
        # Step 2: Reranked Memories
        print_box("STEP 2: Memory Rerank (LLM 재정렬 후)")
        print("LLM Rerank Result:")
        print(f"- Original order: {list(range(len(mp.retrieved_memories)))}")
        print(f"- Retained top {len(mp.reranked_memories)} memories\n")
        print("Top Reranked Memories:\n")
        
        for i, mem in enumerate(mp.reranked_memories, 1):
            print(format_memory_for_display(mem, i, "Rank"))
        
        # Step 3: Rewritten Context
        print_box("STEP 3: Memory Rewrite (최종 Context 변환)")
        print("Rewritten Context for Agent:\n")
        if mp.rewritten_context:
            # Pretty print the rewritten context
            lines = mp.rewritten_context.split('\n')
            for line in lines:
                print(f"  {line}")
        else:
            print("  (No rewritten context available)")
        print()
    
    # Step 4: Agent Execution Trajectory
    print_box("Agent Execution Trajectory (전체 실행 과정)")
    
    for turn in result.trajectory:
        print(f"\n[Turn {turn['step']}]")
        print("  Assistant Code:")
        # Indent the code
        code_lines = turn['assistant_code'].split('\n')
        for line in code_lines[:10]:  # Limit to first 10 lines
            print(f"    {line}")
        if len(code_lines) > 10:
            print(f"    ... ({len(code_lines) - 10} more lines)")
        
        print("\n  Output:")
        output_lines = turn['execution_output'].split('\n')
        for line in output_lines[:10]:  # Limit to first 10 lines
            print(f"    {line}")
        if len(output_lines) > 10:
            print(f"    ... ({len(output_lines) - 10} more lines)")
    
    # Memory Addition Result (if enabled)
    if with_memory and result.added_memories:
        print_box("STEP 4: Memory Addition (새로 추가된 메모리)")
        print(f"Added {len(result.added_memories)} new memories from this execution:\n")
        for i, mem in enumerate(result.added_memories, 1):
            print(format_memory_for_display(mem, i, "New"))
    
    # Memory Update Status
    if with_memory and result.memory_updated:
        print_box("STEP 5: Memory Update (메모리 정보 업데이트)")
        print("Retrieved memories' freq/utility attributes have been updated.")
        if result.score == 1.0:
            print("  → Utility increased (task succeeded)")
        else:
            print("  → Frequency increased (task attempted)")
    
    # Final Result
    print_box("FINAL RESULT")
    print(f"Total Steps: {result.steps}")
    status = "SUCCESS" if result.score == 1.0 else "FAILED"
    print(f"Result: {status} (score: {result.score})")
    
    if with_memory and result.memory_pipeline:
        mp = result.memory_pipeline
        print(f"Memory Contribution: Retrieved {len(mp.retrieved_memories)} → Reranked {len(mp.reranked_memories)}")
        if result.added_memories:
            print(f"Memory Addition: {len(result.added_memories)} new memories added to experience pool")


def save_result_to_json(
    without_memory_result: Optional[ExecutionResult],
    with_memory_result: Optional[ExecutionResult],
    output_path: str
):
    """Save execution results to JSON file for later analysis."""
    
    def trial_to_dict(trial: TrialResult) -> Dict:
        return {
            "run_id": trial.run_id,
            "score": trial.score,
            "steps": trial.steps,
            "task_completed": trial.task_completed,
            "used_previous_memories": trial.used_previous_memories,
            "start_time": trial.start_time,
            "end_time": trial.end_time,
            "trajectory": trial.trajectory,
            "memory_pipeline": {
                "query": trial.memory_pipeline.query if trial.memory_pipeline else "",
                "retrieved_memories": trial.memory_pipeline.retrieved_memories if trial.memory_pipeline else [],
                "reranked_memories": trial.memory_pipeline.reranked_memories if trial.memory_pipeline else [],
                "rewritten_context": trial.memory_pipeline.rewritten_context if trial.memory_pipeline else "",
            } if trial.memory_pipeline else None,
            "added_memories": trial.added_memories,
            "memory_updated": trial.memory_updated,
        }
    
    def result_to_dict(result: ExecutionResult) -> Dict:
        return {
            "task_id": result.task_id,
            "task_instruction": result.task_instruction,
            "score": result.score,
            "steps": result.steps,
            "task_completed": result.task_completed,
            "start_time": result.start_time,
            "end_time": result.end_time,
            "trajectory": result.trajectory,
            "memory_pipeline": {
                "query": result.memory_pipeline.query if result.memory_pipeline else "",
                "retrieved_memories": result.memory_pipeline.retrieved_memories if result.memory_pipeline else [],
                "reranked_memories": result.memory_pipeline.reranked_memories if result.memory_pipeline else [],
                "rewritten_context": result.memory_pipeline.rewritten_context if result.memory_pipeline else "",
            } if result.memory_pipeline else None,
            "added_memories": result.added_memories,
            "memory_updated": result.memory_updated,
            # Failure-aware reflection fields
            "num_trials": result.num_trials,
            "final_run_id": result.final_run_id,
            "trial_results": [trial_to_dict(t) for t in result.trial_results],
        }
    
    output = {
        "task_id": with_memory_result.task_id if with_memory_result else (without_memory_result.task_id if without_memory_result else ""),
        "task_instruction": with_memory_result.task_instruction if with_memory_result else (without_memory_result.task_instruction if without_memory_result else ""),
    }
    
    if without_memory_result:
        output["without_memory"] = result_to_dict(without_memory_result)
    
    if with_memory_result:
        output["with_memory"] = result_to_dict(with_memory_result)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    # Simple test
    task_id = "3d9a636_1"
    
    agent = AppworldReactAgentVerbose(
        task_id=task_id,
        experiment_name="verbose_test",
        use_memory=True,
    )
    
    result = agent.execute()
    print_execution_result(result, with_memory=True)
