"""
Code Iteration Agent for Targeted Code Modification based on User Feedback
Handles iterative code changes using MCP tools, tracking modifications,
and coordinating with Memory Agent for context optimization during iterations.
"""
import json
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
import os
import sys
import re

# Import tiktoken for token calculation (if available)
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

# Import prompts (assuming similar structure to original)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prompts.code_prompts import GENERAL_CODE_ITERATION_SYSTEM_PROMPT

class CodeIterationAgent:
    """
    Code Iteration Agent for targeted, iterative code modification based on user feedback.
    Responsibilities:
    - Execute MCP tool calls for reading/modifying code.
    - Track file modification progress and user feedback incorporation.
    - Monitor iteration status.
    - Coordinate with ConciseMemoryAgent for context optimization during iterations.
    - Calculate token usage for context management during long iterations.
    """
    def __init__(
        self,
        mcp_agent,
        logger: Optional[logging.Logger] = None,
        enable_read_tools: bool = True,
    ):
        """
        Initialize Code Iteration Agent
        Args:
            mcp_agent: MCP agent instance for tool calls
            logger: Logger instance for tracking operations
            enable_read_tools: Whether to enable read_file and read_code_mem tools (default: True)
        """
        self.mcp_agent = mcp_agent
        self.logger = logger or self._create_default_logger()
        self.enable_read_tools = enable_read_tools  # Control read tools execution
        self.iteration_summary = {
            "modified_files": [],
            "user_feedback_addressed": [],
            "technical_decisions": [],
            "important_constraints": [],
            "architecture_notes": [],
            "dependency_analysis": [],
        }
        self.files_modified_count = 0
        self.modified_files_set = set() # Track unique file paths to avoid duplicate counting
        self.files_read_for_iteration = set() # Track files read during iteration
        self.last_summary_iteration_count = 0 # Track the iteration count when last summary was triggered

        # Token calculation settings (similar to original)
        self.max_context_tokens = 200000  # Default max context tokens for Claude-3.5-Sonnet
        self.token_buffer = 10000  # Safety buffer before reaching max
        self.summary_trigger_tokens = self.max_context_tokens - self.token_buffer  # Trigger summary when approaching limit
        self.last_summary_token_count = 0 # Track token count when last summary was triggered

        # Initialize tokenizer
        if TIKTOKEN_AVAILABLE:
            try:
                # Use Claude-3 tokenizer (approximation with OpenAI's o200k_base)
                self.tokenizer = tiktoken.get_encoding("o200k_base")
                self.logger.info("Token calculation enabled with o200k_base encoding")
            except Exception as e:
                self.tokenizer = None
                self.logger.warning(f"Failed to initialize tokenizer: {e}")
        else:
            self.tokenizer = None
            self.logger.warning(
                "tiktoken not available, token-based summary triggering disabled"
            )

        # Memory agent integration
        self.memory_agent = None  # Will be set externally
        self.llm_client = None  # Will be set externally
        self.llm_client_type = None  # Will be set externally

        # Log read tools configuration
        read_tools_status = "ENABLED" if self.enable_read_tools else "DISABLED"
        self.logger.info(
            f"🔧 Code Iteration Agent initialized - Read tools: {read_tools_status}"
        )
        if not self.enable_read_tools:
            self.logger.info(
                "🚫 Testing mode: read_file and read_code_mem will be skipped when called"
            )

    def _create_default_logger(self) -> logging.Logger:
        """Create default logger if none provided"""
        logger = logging.getLogger(f"{__name__}.CodeIterationAgent")
        # Don't add handlers to child loggers - let them propagate to root
        logger.setLevel(logging.INFO)
        return logger

    def get_system_prompt(self) -> str:
        """
        Get the system prompt for code iteration
        """
        return GENERAL_CODE_ITERATION_SYSTEM_PROMPT # Use the iteration-specific prompt

    def set_memory_agent(self, memory_agent, llm_client=None, llm_client_type=None):
        """
        Set memory agent for context optimization during iteration
        Args:
            memory_agent: Memory agent instance
            llm_client: LLM client for optimization (if needed by memory agent)
            llm_client_type: Type of LLM client ("anthropic", "openai", etc.)
        """
        self.memory_agent = memory_agent
        self.llm_client = llm_client
        self.llm_client_type = llm_client_type
        self.logger.info("Memory agent integration configured for iteration")

    async def execute_tool_calls(self, tool_calls: List[Dict]) -> List[Dict]:
        """
        Execute MCP tool calls and track iteration progress
        Args:
            tool_calls: List of tool calls to execute
        Returns:
            List of tool execution results
        """
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_input = tool_call["input"]
            self.logger.info(f"Executing MCP tool for iteration: {tool_name}")
            try:
                # Check if read tools are disabled
                if not self.enable_read_tools and tool_name in [
                    "read_file",
                    "read_code_mem",
                ]:
                    self.logger.info(f"🚫 SKIPPING {tool_name} - Read tools disabled for testing")
                    # Return a mock result indicating the tool was skipped
                    mock_result = json.dumps(
                        {
                            "status": "skipped",
                            "message": f"{tool_name} tool disabled for testing",
                            "tool_disabled": True,
                            "original_input": tool_input,
                        },
                        ensure_ascii=False,
                    )
                    results.append(
                        {
                            "tool_id": tool_call["id"],
                            "tool_name": tool_name,
                            "result": mock_result,
                        }
                    )
                    continue

                # Handle read_file calls (similar logic to original, adapted for iteration)
                if tool_name == "read_file":
                    file_path = tool_call["input"].get("file_path", "unknown")
                    self.logger.info(f"🔍 READ_FILE CALL DETECTED for iteration: {file_path}")
                    self.logger.info(
                        f"📊 Files modified count: {self.files_modified_count}"
                    )
                    self.logger.info(
                        f"🧠 Memory agent available: {self.memory_agent is not None}"
                    )
                    # Optionally, intercept read_file calls if memory agent is available (similar to original)
                    # This depends on your specific iteration logic. For now, we'll execute normally.
                    # if self.memory_agent is not None:
                    #     self.logger.info(f"🔄 INTERCEPTING read_file call for {file_path} (memory agent available)")
                    #     result = await self._handle_read_file_with_memory_optimization(tool_call)
                    #     results.append(result)
                    #     continue
                    # else:
                    #     self.logger.info("📁 NO INTERCEPTION: no memory agent available")

                if self.mcp_agent:
                    # Execute tool call through MCP protocol
                    result = await self.mcp_agent.call_tool(tool_name, tool_input)
                    # Track file modification progress
                    if tool_name == "write_file":
                        await self._track_file_modification(result, tool_call["input"])
                    elif tool_name == "read_file":
                        self._track_dependency_analysis(tool_call, result)

                    results.append(
                        {
                            "tool_id": tool_call["id"],
                            "tool_name": tool_name,
                            "result": result,
                        }
                    )
                else:
                    results.append(
                        {
                            "tool_id": tool_call["id"],
                            "tool_name": tool_name,
                            "result": json.dumps(
                                {
                                    "status": "error",
                                    "message": "MCP agent not initialized",
                                },
                                ensure_ascii=False,
                            ),
                        }
                    )
            except Exception as e:
                self.logger.error(f"MCP tool execution failed: {e}")
                results.append(
                    {
                        "tool_id": tool_call["id"],
                        "tool_name": tool_name,
                        "result": json.dumps(
                            {"status": "error", "message": str(e)}, ensure_ascii=False
                        ),
                    }
                )
        return results

    async def _track_file_modification(self, result: Any, input_data: Dict):
        """
        Track file modification progress based on write_file result.
        Args:
            result: The result of the write_file tool call.
            input_data: The input data for the write_file call (contains file_path).
        """
        try:
            # Handle different result types from MCP (similar to original)
            result_data = None
            if hasattr(result, "content"):
                if hasattr(result.content, "text"):
                    result_content = result.content.text
                else:
                    result_content = str(result.content)
                try:
                    result_data = json.loads(result_content)
                except json.JSONDecodeError:
                    result_data = {
                        "status": "success",
                        "file_path": input_data.get("file_path", "unknown"),
                    }
            elif isinstance(result, str):
                try:
                    result_data = json.loads(result)
                except json.JSONDecodeError:
                    result_data = {
                        "status": "success",
                        "file_path": input_data.get("file_path", "unknown"),
                    }
            elif isinstance(result, dict):
                result_data = result
            else:
                result_data = {
                    "status": "success",
                    "file_path": input_data.get("file_path", "unknown"),
                }

            file_path = None
            if result_data and result_data.get("status") == "success":
                file_path = result_data.get("file_path", input_data.get("file_path", "unknown"))
            else:
                file_path = input_data.get("file_path")

            # Only count unique files modified, not repeated tool calls on same file
            if file_path and file_path not in self.modified_files_set:
                self.modified_files_set.add(file_path)
                self.files_modified_count += 1
                self.logger.info(
                    f"New file modification tracked: count={self.files_modified_count}, file={file_path}"
                )
                # Add to modified files list
                self.iteration_summary["modified_files"].append(
                    {
                        "file": file_path,
                        "iteration": self.files_modified_count,
                        "timestamp": time.time(),
                        "size": result_data.get("size", 0) if result_data else 0,
                    }
                )
                # Optionally, trigger memory optimization or summary creation here if needed by the memory agent
                if self.memory_agent:
                    try:
                        # Example: Notify memory agent about the modification
                        await self.memory_agent.record_file_modification(file_path, self.files_modified_count)
                    except Exception as e:
                        self.logger.error(f"Failed to notify memory agent about modification: {e}")
            elif file_path and file_path in self.modified_files_set:
                self.logger.debug(
                    f"File already tracked for modification, skipping duplicate count: {file_path}"
                )
            else:
                self.logger.warning("No valid file path found for modification tracking")

        except Exception as e:
            self.logger.warning(f"Failed to track file modification: {e}")
            # Even if tracking fails, try to count based on tool input (but check for duplicates)
            file_path = input_data.get("file_path")
            if file_path and file_path not in self.modified_files_set:
                self.modified_files_set.add(file_path)
                self.files_modified_count += 1
                self.logger.info(
                    f"File modification counted (emergency fallback): count={self.files_modified_count}, file={file_path}"
                )

    def _track_dependency_analysis(self, tool_call: Dict, result: Any):
        """
        Track dependency analysis through read_file calls during iteration
        (Logic similar to original _track_dependency_analysis)
        """
        try:
            file_path = tool_call["input"].get("file_path")
            if file_path:
                # Track unique files read for dependency analysis during iteration
                if file_path not in self.files_read_for_iteration:
                    self.files_read_for_iteration.add(file_path)
                    # Add to dependency analysis summary
                    self.iteration_summary["dependency_analysis"].append(
                        {
                            "file_read": file_path,
                            "timestamp": time.time(),
                            "purpose": "iteration_dependency_analysis",
                        }
                    )
                    self.logger.info(
                        f"Iteration dependency analysis tracked: file_read={file_path}"
                    )
        except Exception as e:
            self.logger.warning(f"Failed to track iteration dependency analysis: {e}")

    def calculate_messages_token_count(self, messages: List[Dict]) -> int:
        """
        Calculate total token count for a list of messages
        (Copied from original CodeImplementationAgent)
        """
        if not self.tokenizer:
            # Fallback: rough estimation based on character count
            total_chars = sum(len(str(msg.get("content", ""))) for msg in messages)
            # Rough approximation: 1 token ≈ 4 characters
            return total_chars // 4

        try:
            total_tokens = 0
            for message in messages:
                content = str(message.get("content", ""))
                role = message.get("role", "")
                # Count tokens for content
                if content:
                    content_tokens = len(
                        self.tokenizer.encode(content, disallowed_special=())
                    )
                    total_tokens += content_tokens
                # Add tokens for role and message structure
                role_tokens = len(self.tokenizer.encode(role, disallowed_special=()))
                total_tokens += role_tokens + 4  # Extra tokens for message formatting
            return total_tokens
        except Exception as e:
            self.logger.warning(f"Token calculation failed: {e}")
            # Fallback estimation
            total_chars = sum(len(str(msg.get("content", ""))) for msg in messages)
            return total_chars // 4

    def should_trigger_summary_by_tokens(self, messages: List[Dict]) -> bool:
        """
        Check if summary should be triggered based on token count
        (Copied from original CodeImplementationAgent)
        """
        if not messages:
            return False
        # Calculate current token count
        current_token_count = self.calculate_messages_token_count(messages)
        # Check if we should trigger summary
        should_trigger = (
            current_token_count > self.summary_trigger_tokens
            and current_token_count
            > self.last_summary_token_count
            + 10000  # Minimum 10k tokens between summaries
        )
        if should_trigger:
            self.logger.info(
                f"Token-based summary trigger for iteration: current={current_token_count:,}, "
                f"threshold={self.summary_trigger_tokens:,}, "
                f"last_summary={self.last_summary_token_count:,}"
            )
        return should_trigger

    def should_trigger_summary(
        self, summary_trigger: int = 5, messages: List[Dict] = None
    ) -> bool:
        """
        Check if summary should be triggered based on token count (preferred) or modification count (fallback)
        Args:
            summary_trigger: Number of files after which to trigger summary (fallback)
            messages: Current conversation messages for token calculation
        Returns:
            True if summary should be triggered
        """
        # Primary: Token-based triggering
        if messages and self.tokenizer:
            return self.should_trigger_summary_by_tokens(messages)

        # Fallback: Modification-based triggering (similar to original file-based)
        self.logger.info("Using fallback modification-based summary triggering for iteration")
        should_trigger = (
            self.files_modified_count > 0
            and self.files_modified_count % summary_trigger == 0
            and self.files_modified_count > self.last_summary_iteration_count
        )
        return should_trigger

    def mark_summary_triggered(self, messages: List[Dict] = None):
        """
        Mark that summary has been triggered for current state
        Args:
            messages: Current conversation messages for token tracking
        """
        # Update modification-based tracking
        self.last_summary_iteration_count = self.files_modified_count
        # Update token-based tracking
        if messages and self.tokenizer:
            self.last_summary_token_count = self.calculate_messages_token_count(
                messages
            )
            self.logger.info(
                f"Iteration summary marked as triggered - modification_count: {self.files_modified_count}, "
                f"token_count: {self.last_summary_token_count:,}"
            )
        else:
            self.logger.info(
                f"Iteration summary marked as triggered for modification count: {self.files_modified_count}"
            )

    def get_iteration_summary(self) -> Dict[str, Any]:
        """
        Get current iteration summary
        """
        return self.iteration_summary.copy()

    def get_files_modified_count(self) -> int:
        """
        Get the number of files modified so far
        """
        return self.files_modified_count

    def get_read_tools_status(self) -> Dict[str, Any]:
        """
        Get read tools configuration status
        Returns:
            Dictionary with read tools status information
        """
        return {
            "read_tools_enabled": self.enable_read_tools,
            "status": "ENABLED" if self.enable_read_tools else "DISABLED",
            "tools_affected": ["read_file", "read_code_mem"],
            "description": "Read tools configuration for iteration testing purposes",
        }

    def add_technical_decision(self, decision: str, context: str = ""):
        """
        Add a technical decision to the iteration summary
        Args:
            decision: Description of the technical decision
            context: Additional context for the decision
        """
        self.iteration_summary["technical_decisions"].append(
            {"decision": decision, "context": context, "timestamp": time.time()}
        )
        self.logger.info(f"Technical decision recorded for iteration: {decision}")

    def add_constraint(self, constraint: str, impact: str = ""):
        """
        Add an important constraint to the iteration summary
        Args:
            constraint: Description of the constraint
            impact: Impact of the constraint on iteration
        """
        self.iteration_summary["important_constraints"].append(
            {"constraint": constraint, "impact": impact, "timestamp": time.time()}
        )
        self.logger.info(f"Constraint recorded for iteration: {constraint}")

    def add_architecture_note(self, note: str, component: str = ""):
        """
        Add an architecture note to the iteration summary
        Args:
            note: Architecture note description
            component: Related component or module
        """
        self.iteration_summary["architecture_notes"].append(
            {"note": note, "component": component, "timestamp": time.time()}
        )
        self.logger.info(f"Architecture note recorded for iteration: {note}")

    def get_iteration_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive iteration statistics
        """
        return {
            "total_files_modified": self.files_modified_count,
            "files_modified_count": self.files_modified_count,
            "technical_decisions_count": len(
                self.iteration_summary["technical_decisions"]
            ),
            "constraints_count": len(
                self.iteration_summary["important_constraints"]
            ),
            "architecture_notes_count": len(
                self.iteration_summary["architecture_notes"]
            ),
            "dependency_analysis_count": len(
                self.iteration_summary["dependency_analysis"]
            ),
            "files_read_for_iteration": len(self.files_read_for_iteration),
            "unique_files_modified": len(self.modified_files_set),
            "modified_files_list": [
                f["file"] for f in self.iteration_summary["modified_files"]
            ],
            "dependency_files_read": list(self.files_read_for_iteration),
            "last_summary_iteration_count": self.last_summary_iteration_count,
            "read_tools_status": self.get_read_tools_status(),
        }

    def reset_iteration_tracking(self):
        """
        Reset iteration tracking (useful for new iteration sessions)
        """
        self.iteration_summary = {
            "modified_files": [],
            "user_feedback_addressed": [],
            "technical_decisions": [],
            "important_constraints": [],
            "architecture_notes": [],
            "dependency_analysis": [],
        }
        self.files_modified_count = 0
        self.modified_files_set = set()
        self.files_read_for_iteration = set()
        self.last_summary_iteration_count = 0
        self.last_summary_token_count = 0
        self.logger.info("Iteration tracking reset")