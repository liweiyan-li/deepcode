"""
Code Iteration Workflow - MCP-compliant Iterative Modification based on User Feedback
Features:
1. Iterative Code Modification based on user intent
2. Context Optimization using Memory Agent
3. MCP Architecture integration (similar to original)
"""
import re
import asyncio
import json
import logging
import os
import sys
import time
import yaml
import shutil
from typing import Dict, Any, Optional, List

# MCP Agent imports (similar to original)
from mcp_agent.agents.agent import Agent

# Local imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prompts.code_prompts import (
    GENERAL_CODE_ITERATION_SYSTEM_PROMPT, 
)

# Import agents
from workflows.agents.code_iteration_agent import CodeIterationAgent # Import the new agent
from workflows.agents.memory_agent_concise import ConciseMemoryAgent # Assuming this is available

# Import utility functions from the original workflow if needed
from config.mcp_tool_definitions import get_mcp_tools # Assuming this is available
from utils.llm_utils import get_preferred_llm_class, get_default_models # Assuming this is available

# Import original workflow functions if needed (e.g., _validate_messages, _call_llm_with_tools)
# For this example, we'll assume they are defined within this class or imported separately
# from workflows.code_implementation_workflow import _validate_messages, _call_llm_with_tools # Example

class CodeIterationWorkflow:
    """
    Code Iteration Workflow Manager
    Uses standard MCP architecture:
    1. Connect to code-implementation server via MCP client (or a specific iteration server if needed)
    2. Use MCP protocol for tool calls (read_file, write_file, etc.)
    3. Support workspace management and operation history tracking
    4. Iteratively modify code based on user feedback
    """
    # ==================== 1. Class Initialization and Configuration (Infrastructure Layer) ====================
    def __init__(self, config_path: str = "mcp_agent.secrets.yaml"):
        """Initialize workflow with configuration"""
        self.config_path = config_path
        self.api_config = self._load_api_config()
        self.default_models = get_default_models("mcp_agent.config.yaml")
        self.logger = self._create_logger()
        self.mcp_agent = None
        self.enable_read_tools = True # Default value, will be overridden by run_workflow parameter

    def _load_api_config(self) -> Dict[str, Any]:
        """Load API configuration from YAML file (Copied from original)"""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise Exception(f"Failed to load API config: {e}")

    def _create_logger(self) -> logging.Logger:
        """Create and configure logger (Copied from original)"""
        logger = logging.getLogger(__name__)
        # Don't add handlers to child loggers - let them propagate to root
        logger.setLevel(logging.INFO)
        return logger

    def _validate_messages(self, messages: List[Dict]) -> List[Dict]:
        """Validate and clean message list (Copied from original)"""
        valid_messages = []
        for msg in messages:
            content = msg.get("content", "").strip()
            if content:
                valid_messages.append(
                    {"role": msg.get("role", "user"), "content": content}
                )
            else:
                self.logger.warning(f"Skipping empty message: {msg}")
        return valid_messages

    async def _call_llm_with_tools(
        self, client, client_type, system_message, messages, tools, max_tokens=8192
    ):
        """
        Call LLM with tools (Copied from original workflow).
        This function contains the complex logic for calling Anthropic, OpenAI, Google APIs.
        """
        # --- Start of Copied Block ---
        try:
            if client_type == "openai":
                print("尝试调用工具")
                return await self._call_openai_with_tools(
                    client, system_message, messages, tools, max_tokens
                )
            else:
                raise ValueError(f"Unsupported client type: {client_type}")
        except Exception as e:
            self.logger.error(f"LLM call failed: {e}")
            raise
        # --- End of Copied Block ---

    async def _call_openai_with_tools(
        self, client, system_message, messages, tools, max_tokens
    ):
        """Call OpenAI API with robust JSON error handling and retry mechanism (Copied from original)"""
        openai_tools = []
        for tool in tools:
            openai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["input_schema"],
                    },
                }
            )

        openai_messages = [{"role": "system", "content": system_message}]
        openai_messages.extend(messages)

        # Retry mechanism for API calls
        max_retries = 3
        retry_delay = 2  # seconds
        for attempt in range(max_retries):
            try:
                # Try max_tokens first, fallback to max_completion_tokens if unsupported
                try:
                    response = await client.chat.completions.create(
                        model=self.default_models["openai"],
                        messages=openai_messages,
                        tools=openai_tools if openai_tools else None,
                        max_tokens=max_tokens,
                        temperature=0.2,
                    )
                except Exception as e:
                    if "max_tokens" in str(e) and "max_completion_tokens" in str(e):
                        # Retry with max_completion_tokens for models that require it
                        response = await client.chat.completions.create(
                            model=self.default_models["openai"],
                            messages=openai_messages,
                            tools=openai_tools if openai_tools else None,
                            max_completion_tokens=max_tokens,
                        )
                    else:
                        raise

                # Validate response structure
                if (
                    not response
                    or not hasattr(response, "choices")
                    or not response.choices
                ):
                    raise ValueError("Invalid API response: missing choices")

                if not response.choices[0] or not hasattr(
                    response.choices[0], "message"
                ):
                    raise ValueError("Invalid API response: missing message in choice")

                message = response.choices[0].message
                content = message.content or ""

                # Successfully got a valid response
                break
            except json.JSONDecodeError as e:
                print(
                    f"❌ JSON Decode Error in API response (attempt {attempt + 1}/{max_retries}):"
                )
                print(f"   Error: {e}")
                print(f"   Position: line {e.lineno}, column {e.colno}")
                if attempt < max_retries - 1:
                    print(f"   ⏳ Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    print("   ❌ All retries exhausted")
                    raise
            except (ValueError, AttributeError, TypeError) as e:
                print(f"❌ API Response Error (attempt {attempt + 1}/{max_retries}):")
                print(f"   Error type: {type(e).__name__}")
                print(f"   Error: {e}")
                if attempt < max_retries - 1:
                    print(f"   ⏳ Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    print("   ❌ All retries exhausted")
                    # Return empty response instead of crashing
                    return {
                        "content": "API error - unable to get valid response",
                        "tool_calls": [],
                    }
            except Exception as e:
                print(
                    f"❌ Unexpected API Error (attempt {attempt + 1}/{max_retries}):"
                )
                print(f"   Error type: {type(e).__name__}")
                print(f"   Error: {e}")
                if attempt < max_retries - 1:
                    print(f"   ⏳ Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    print("   ❌ All retries exhausted")
                    raise

        tool_calls = []
        if message.tool_calls:
            for tool_call in message.tool_calls:
                try:
                    # Attempt to parse tool call arguments
                    parsed_input = json.loads(tool_call.function.arguments)
                    tool_calls.append(
                        {
                            "id": tool_call.id,
                            "name": tool_call.function.name,
                            "input": parsed_input,
                        }
                    )
                except json.JSONDecodeError as e:
                    # Detailed JSON parsing error logging
                    print("❌ JSON Parsing Error in tool call:")
                    print(f"   Tool: {tool_call.function.name}")
                    print(f"   Error: {e}")
                    print("   Raw arguments (first 500 chars):")
                    print(f"   {tool_call.function.arguments[:500]}")
                    print(f"   Error position: line {e.lineno}, column {e.colno}")
                    print(
                        f"   Problem at: ...{tool_call.function.arguments[max(0, e.pos-50):e.pos+50]}..."
                    )
                    # Attempt advanced JSON repair (assuming _repair_truncated_json is defined)
                    # repaired = self._repair_truncated_json(
                    #     tool_call.function.arguments, tool_call.function.name
                    # )
                    # if repaired:
                    #     print("   ✅ JSON repaired successfully")
                    #     tool_calls.append(
                    #         {
                    #             "id": tool_call.id,
                    #             "name": tool_call.function.name,
                    #             "input": repaired,
                    #         }
                    #     )
                    # else:
                    #     # Skip this tool call if repair failed
                    #     print("   ⚠️  Skipping unrepairable tool call")
                    #     continue
                    # For simplicity, re-raise the error here if repair function is not available
                    raise e

        return {"content": content, "tool_calls": tool_calls}

    async def _fallback_apply_patch_from_raw(self, raw_content: str, iteration_dir: str) -> Dict:
        """
        Try to extract a JSON mapping or code-blocks from raw LLM output and call write_multiple_files.
        Returns the tool call result or {"status":"no_patch_found"}.
        """
        # 1) 找 JSON 对象或字典结构 mapping path -> content
        # 尝试直接找花括号形式的大 JSON
        json_match = re.search(r"(\{\\s*\"?.*?\\}\\s*)", raw_content, re.DOTALL)
        if json_match:
            candidate = json_match.group(1)
            # 继续尝试解析为 dict[str,str]
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in parsed.items()):
                    payload = json.dumps(parsed, ensure_ascii=False)
                    # 调用 MCP 写多个文件
                    try:
                        res = await self.mcp_agent.call_tool("write_multiple_files", {"file_implementations": payload, "create_backup": True})
                        return {"status": "called_write_multiple_files", "result": res}
                    except Exception as e:
                        return {"status": "write_call_failed", "error": str(e)}
            except Exception:
                pass

        # 2) 如果没有 JSON，再尝试按自定义分隔符：例如 LLM 可能输出 === FILE: path ===\n<code>
        blocks = re.split(r"={3,}\s*FILE:\s*(.*?)\s*={3,}", raw_content)
        # split 会产生交替 list (prefix, path1, content1, path2, content2, ...)
        if len(blocks) >= 3:
            files = {}
            # 入块解析（偶数位是路径，奇数位是内容，这依赖具体 split 行为）
            for i in range(1, len(blocks), 2):
                path = blocks[i].strip()
                content = blocks[i+1] if i+1 < len(blocks) else ""
                if path and content:
                    files[path] = content.strip()
            if files:
                try:
                    payload = json.dumps(files, ensure_ascii=False)
                    res = await self.mcp_agent.call_tool("write_multiple_files", {"file_implementations": payload, "create_backup": True})
                    return {"status": "called_write_multiple_files", "result": res}
                except Exception as e:
                    return {"status": "write_call_failed", "error": str(e)}

        return {"status": "no_patch_found"}
    # ==================== 2. Public Interface Methods (External API Layer) ====================
    async def run_iteration(
        self,
        user_intent: str, # User's modification request
        target_directory: str, # The directory containing the code to modify
        original_code_dir: str = "generate_code",
        iteration_dir_name: Optional[str] = None,
        initial_code_snapshot: str = None, # Optional: Initial code context (e.g., from first implementation)
        test_report_before: str = None, # Optional: Test report before iteration
        max_iterations: int = 5, # Maximum number of iteration loops
        enable_read_tools: bool = True, # Whether to enable read tools
    ):
        logging.info("开始执行代码迭代工作流")
        
        """
        Run complete iteration workflow - Main public interface for modifying existing code.
        Args:
            user_intent: The user's modification request (e.g., "Fix the login bug", "Add feature X")
            target_directory: The directory containing the existing codebase to modify.
            initial_code_snapshot: Optional initial context (e.g., first N files from initial implementation).
            test_report_before: Optional test report from before the iteration starts.
            max_iterations: Maximum number of LLM interaction loops.
            enable_read_tools: Whether to enable read_file and read_code_mem tools.
        """
        # Set the read tools configuration
        self.enable_read_tools = enable_read_tools
        try:
            self.logger.info("=" * 80)
            self.logger.info("🔄 STARTING CODE ITERATION WORKFLOW (NEW FOLDER MODE)")
            self.logger.info("=" * 80)
            self.logger.info(f"🎯 User Intent: {user_intent}")
            self.logger.info(f"📂 Target Directory: {target_directory}")
            self.logger.info(f"📁 Original Code Dir: {original_code_dir}")
            self.logger.info(f"🆕 Iteration Dir Name: {iteration_dir_name or 'auto-generated'}")
            self.logger.info(f"📊 Initial Snapshot Provided: {initial_code_snapshot is not None}")
            self.logger.info(f"🧪 Test Report Before: {test_report_before is not None}")
            self.logger.info(f"⚙️  Max Iterations: {max_iterations}")
            self.logger.info(
                f"🔧 Read tools: {'ENABLED' if self.enable_read_tools else 'DISABLED'}"
            )
            self.logger.info("=" * 80)

            original_code_path = os.path.join(target_directory, original_code_dir)
            if not os.path.exists(original_code_path):
                raise FileNotFoundError(f"Original code directory not found: {original_code_path}")

            if iteration_dir_name is None:
                 # Auto-generate a name based on user intent or timestamp
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                # Sanitize user intent for use in path
                sanitized_intent = "".join(c if c.isalnum() else "_" for c in user_intent[:20])
                iteration_dir_name = f"iteration_{sanitized_intent}_{timestamp}"

            iteration_directory = os.path.join(target_directory, iteration_dir_name)

            if os.path.exists(iteration_directory):
                self.logger.warning(f"Iteration directory already exists: {iteration_directory}. Overwriting contents.")
                shutil.rmtree(iteration_directory)

            os.makedirs(iteration_directory, exist_ok=True)
            self.logger.info(f"📁 Created new iteration directory: {iteration_directory}")

            # Copy the original code into the new iteration directory
            self.logger.info(f"🔄 Copying original code from '{original_code_path}' to '{iteration_directory}'...")
            shutil.copytree(original_code_path, iteration_directory, dirs_exist_ok=True) # dirs_exist_ok=True is crucial here as the target dir was just created
            self.logger.info(f"✅ Original code copied successfully to '{iteration_directory}'.")

            client, client_type = await self._initialize_llm_client()
            await self._initialize_mcp_agent(iteration_directory) # Set workspace to target dir
            tools = self._prepare_mcp_tool_definitions()

            # Initialize specialized agents
            iteration_agent = CodeIterationAgent(
                self.mcp_agent, self.logger, self.enable_read_tools
            )

            # Initialize memory agent (for context optimization during iteration)
            memory_agent = ConciseMemoryAgent(
                initial_plan_content=user_intent, # Use user intent as the "plan" for iteration
                logger=self.logger,
                target_directory=target_directory,
                default_models=self.default_models,
                code_directory=iteration_directory, # Code dir is target dir for iteration
            )
            print("初始化 memory agent 完成")
            # assert False
            # Connect iteration agent with memory agent
            iteration_agent.set_memory_agent(memory_agent, client, client_type)

            # Prepare initial system message and messages
            system_message = GENERAL_CODE_ITERATION_SYSTEM_PROMPT # Use the iteration-specific prompt
            messages = []

            # Build the initial iteration message (context)
            iteration_message = self._build_iteration_context(
                user_intent=user_intent,
                initial_code_snapshot=initial_code_snapshot,
                test_report_before=test_report_before,
                memory_agent=memory_agent,
                iteration=1,
                iteration_directory_name=iteration_dir_name 
            )
            messages.append({"role": "user", "content": iteration_message})

            # Run the iteration loop
            result = await self._code_iteration_loop(
                client=client,
                client_type=client_type,
                system_message=system_message,
                messages=messages,
                tools=tools,
                user_intent=user_intent,
                target_directory=target_directory,
                iteration_directory=iteration_directory,
                iteration_agent=iteration_agent,
                memory_agent=memory_agent,
                max_iterations=max_iterations
            )

            if iteration_agent.get_files_modified_count() == 0:
                self.logger.info("🔧 There is nothing in the code that needs to be changed")
            #     # 针对 lora_merger.py 插入注释
            #     file_path = os.path.join(iteration_directory, "lora_merger.py")
            #     try:
            #         read_res = await self.mcp_agent.call_tool("read_file", {"file_path": file_path})
            #         # 解析返回的 JSON 格式结果
            #         if isinstance(read_res, str):
            #             read_data = json.loads(read_res)
            #         else:
            #             content_text = read_res.content.text if hasattr(read_res.content, "text") else str(read_res)
            #             read_data = json.loads(content_text)
            #         original_content = read_data.get("content", "")
            #     except Exception as e:
            #         self.logger.warning(f"Failed to read file for patching: {e}")
            #         original_content = ""
            #     patched_content = original_content + "\n# Minimal patch added for testing purposes\n"
            #     write_res = await self.mcp_agent.call_tool("write_file", {"file_path": file_path, "content": patched_content})
            #     # 跟踪文件修改，更新统计
            #     await iteration_agent._track_file_modification(write_res, {"file_path": file_path})
            
            self.logger.info("Iteration workflow execution successful")
            return {
                "status": "success",
                "user_intent": user_intent,
                "target_directory": target_directory,
                "results": result,
                "mcp_architecture": "standard",
            }

        except Exception as e:
            self.logger.error(f"Iteration workflow execution failed: {e}")
            return {"status": "error", "message": str(e), "user_intent": user_intent}
        finally:
            await self._cleanup_mcp_agent()

    def _build_iteration_context(
        self,
        user_intent: str,
        initial_code_snapshot: str,
        test_report_before: str,
        memory_agent: Any,
        iteration: int,
        iteration_directory_name: str
    ) -> str:
        """Build the initial context message for the iteration loop."""
        # Get context from memory agent (if available)
        context_summary = memory_agent.get_current_context() if hasattr(memory_agent, 'get_current_context') else "No specific context available."

        return f"""**Task: Perform targeted code modifications based on user feedback (Iteration {iteration})**

**User Modification Request:**
{user_intent}

**Project Context Summary:**
{context_summary}

**Critical Test Failures (Before Iteration):**
{test_report_before or 'No test report provided'}
**Working Directory:** {self.mcp_agent.workspace_path if self.mcp_agent else 'Not set'}
**Iteration Directory Name:** {iteration_directory_name}
**Note:** The original code from 'generate_code' has been copied into the '{iteration_directory_name}' folder. Files are now located relative to this new workspace. For example, if the original file was 'generate_code/from_reproduction/src/merging/lora_merger.py', its path in this iteration is 'from_reproduction/src/merging/lora_merger.py'.

**Current Action Plan:**
1. **ANALYZE** the modification request against the provided context.
2. **VALIDATE** context BEFORE making changes (use `read_file` for necessary files).
3. **EXECUTE** minimal changes to address the request.
4. **VERIFY** changes by re-running relevant tests if possible.

**Next Step:** Begin with the files most relevant to the user request. Use `read_file` to inspect, `write_file` to modify. All operations are isolated within the '{iteration_directory_name}' folder."""

    # ==================== 3. Core Business Logic (Implementation Layer) ====================
    async def _code_iteration_loop(
        self,
        client,
        client_type,
        system_message,
        messages,
        tools,
        user_intent,
        target_directory,
        iteration_directory,
        iteration_agent,
        memory_agent,
        max_iterations,
    ):
        """Main loop for iterative code modification."""
        iteration = 0
        start_time = time.time()
        max_time = 1200 # 20 minutes time limit

        while iteration < max_iterations:
            iteration += 1
            elapsed_time = time.time() - start_time

            if elapsed_time > max_time:
                self.logger.warning(f"Time limit reached: {elapsed_time:.2f}s")
                break

            self.logger.info(f"Code iteration loop iteration {iteration}")

            messages = self._validate_messages(messages)
            print("2")
            current_system_message = iteration_agent.get_system_prompt()
            print("1")
            print(messages)
            # Call LLM
            response = await self._call_llm_with_tools(
                client, client_type, current_system_message, messages, tools
            )
            try:
                tools_list = await self.mcp_agent.list_tools()
                # tools_list 的结构依赖 mcp impl，直接记录原样即可
                self.logger.info(f"Available MCP tools: {tools_list}")
            except Exception as e:
                self.logger.warning(f"Failed to list MCP tools: {e}")
            print("=============================")
            import pprint
            pprint.pprint(response)

            # 更友好地提取并打印关键字段
            response_content = response.get("content", "") if isinstance(response, dict) else str(response)
            print(">>> LLM content:\n", response_content[:2000])
            response_content = response.get("content", "").strip()
            if not response_content:
                response_content = f"Continue iterating on the modification request: {user_intent}"

            messages.append({"role": "assistant", "content": response_content})

            # Handle tool calls
            if response.get("tool_calls"):
                tool_results = await iteration_agent.execute_tool_calls(
                    response["tool_calls"]
                )

                # Record essential tool results in concise memory agent
                for tool_call, tool_result in zip(response["tool_calls"], tool_results):
                    memory_agent.record_tool_result(
                        tool_name=tool_call["name"],
                        tool_input=tool_call["input"],
                        tool_result=tool_result.get("result"),
                    )

                # Determine guidance based on results
                has_error = self._check_tool_results_for_errors(tool_results)
                files_modified_count = iteration_agent.get_files_modified_count()

                if has_error:
                    guidance = self._generate_error_guidance()
                else:
                    guidance = self._generate_success_guidance(files_modified_count)

                compiled_response = self._compile_user_response(tool_results, guidance)
                messages.append({"role": "user", "content": compiled_response})

                # NEW LOGIC: Apply memory optimization immediately after write_file detection (if applicable)
                if memory_agent.should_trigger_memory_optimization(
                    messages, iteration_agent.get_files_modified_count()
                ):
                    self.logger.info("Memory optimization triggered by memory agent.")
                    current_system_message = iteration_agent.get_system_prompt()
                    messages = memory_agent.apply_memory_optimization(
                        current_system_message, messages, iteration_agent.get_files_modified_count()
                    )

            else:
                files_modified_count = iteration_agent.get_files_modified_count()
                no_tools_guidance = self._generate_no_tools_guidance(files_modified_count)
                messages.append({"role": "user", "content": no_tools_guidance})

            # Record file modifications in memory agent (for the current round)
            for file_info in iteration_agent.get_iteration_summary()["modified_files"]:
                memory_agent.record_file_implementation(file_info["file"]) # Reuse method name if structure is compatible

            # Start new round for next iteration, sync with workflow iteration
            memory_agent.start_new_round(iteration=iteration)

            # Check completion based on iteration count or other criteria (e.g., no more changes needed)
            # For now, we rely on max_iterations and time limit.
            # You can add more sophisticated completion checks here.

            # Emergency trim if too long
            if len(messages) > 50:
                self.logger.warning(
                    "Emergency message trim - applying concise memory optimization"
                )
                current_system_message = iteration_agent.get_system_prompt()
                files_modified_count = iteration_agent.get_files_modified_count()
                messages = memory_agent.apply_memory_optimization(
                    current_system_message, messages, files_modified_count
                )

        return await self._generate_iteration_final_report(
            iteration, time.time() - start_time, iteration_agent, memory_agent, iteration_directory,
        )

    # ==================== 4. MCP Agent and LLM Communication Management (Communication Layer) ====================
    # (Copied from original, with minor adaptation for target_directory)
    async def _initialize_mcp_agent(self, iteration_directory: str):
        """Initialize MCP agent and connect to code-implementation server"""
        try:
            # Use the same server names as the original implementation workflow, or a specific iteration server
            self.mcp_agent = Agent(
                name="CodeIterationAgent",
                instruction="You are a code iteration assistant, using MCP tools to modify existing code based on user feedback.",
                server_names=["code-iteration"], 
            )
            await self.mcp_agent.__aenter__()
            llm = await self.mcp_agent.attach_llm(
                get_preferred_llm_class(self.config_path)
            )
            # Set workspace to the target code directory
            workspace_result = await self.mcp_agent.call_tool(
                "set_workspace", {"workspace_path": iteration_directory}
            )
            self.mcp_agent.workspace_path = iteration_directory
            self.logger.info(f"Workspace setup result: {workspace_result}")
            return llm
        except Exception as e:
            self.logger.error(f"Failed to initialize MCP agent: {e}")
            if self.mcp_agent:
                try:
                    await self.mcp_agent.__aexit__(None, None, None)
                except Exception:
                    pass
                self.mcp_agent = None
            raise

    async def _cleanup_mcp_agent(self):
        """Clean up MCP agent resources (Copied from original)"""
        if self.mcp_agent:
            try:
                await self.mcp_agent.__aexit__(None, None, None)
                self.logger.info("MCP agent connection closed")
            except Exception as e:
                self.logger.warning(f"Error closing MCP agent: {e}")
            finally:
                self.mcp_agent = None

    async def _initialize_llm_client(self):
        """Initialize LLM client (Copied from original)"""
        # --- Start of Copied Block ---
        # Get API keys
        anthropic_key = self.api_config.get("anthropic", {}).get("api_key", "")
        openai_key = self.api_config.get("openai", {}).get("api_key", "")
        
        print(f"已获取{openai_key}")
        google_key = self.api_config.get("google", {}).get("api_key", "")

        # Read user preference from main config
        preferred_provider = None
        try:
            import yaml
            config_path = "mcp_agent.config.yaml"
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)
                    preferred_provider = config.get("llm_provider", "").strip().lower()
        except Exception as e:
            self.logger.warning(f"Could not read llm_provider preference: {e}")

        async def init_openai():
            if not (openai_key and openai_key.strip()):
                return None
            try:
                from openai import AsyncOpenAI
                openai_config = self.api_config.get("openai", {})
                base_url = openai_config.get("base_url")
                if base_url:
                    client = AsyncOpenAI(api_key=openai_key, base_url=base_url)
                else:
                    client = AsyncOpenAI(api_key=openai_key)
                model_name = self.default_models.get("openai", "o3-mini")
                print(model_name)
                try:
                    await client.chat.completions.create(
                        model=model_name,
                        max_completion_tokens=20,
                        messages=[{"role": "user", "content": "test"}],
                    )
                except Exception as e:
                    if "max_tokens" in str(e) and "max_completion_tokens" in str(e):
                        self.logger.info(
                            f"Model {model_name} requires max_completion_tokens parameter"
                        )
                        await client.chat.completions.create(
                            model=model_name,
                            max_completion_tokens=20,
                            messages=[{"role": "user", "content": "test"}],
                        )
                    else:
                        raise
                self.logger.info(f"Using OpenAI API with model: {model_name}")
                if base_url:
                    self.logger.info(f"Using custom base URL: {base_url}")
                return client, "openai"
            except Exception as e:
                self.logger.warning(f"OpenAI API unavailable: {e}")
                return None

        # Map providers to their init functions
        provider_init_map = {
            "openai": init_openai,
        }

        # Try preferred provider first
        if preferred_provider and preferred_provider in provider_init_map:
            self.logger.info(f"🎯 Trying preferred provider: {preferred_provider}")
            result = await provider_init_map[preferred_provider]()
            if result:
                return result
            else:
                self.logger.warning(
                    f"⚠️ Preferred provider '{preferred_provider}' unavailable, trying alternatives..."
                )

        # Fallback: try providers in order
        for provider_name, init_func in provider_init_map.items():
            if provider_name == preferred_provider:
                continue  # Already tried
            result = await init_func()
            if result:
                return result

        raise ValueError(
            "No available LLM API - please check your API keys in configuration"
        )
        # --- End of Copied Block ---


    # ==================== 5. Tools and Utility Methods (Utility Layer) ====================
    def _prepare_mcp_tool_definitions(self) -> List[Dict[str, Any]]:
        """Prepare tool definitions in Anthropic API standard format (Copied from original)"""
        return get_mcp_tools("code_implementation") # Reuse tool definitions

    def _check_tool_results_for_errors(self, tool_results: List[Dict]) -> bool:
        """Check tool results for errors (Copied from original, adapted for iteration agent)"""
        # This function can be copied directly from the original workflow
        # It checks the 'result' field of tool results for error indicators.
        # Assuming the structure of tool results is similar.
        for result in tool_results:
            try:
                # Example: Check if result contains error keywords or status
                result_content = result.get("result", "")
                if isinstance(result_content, str):
                    if "error" in result_content.lower() or "failed" in result_content.lower():
                        return True
                elif isinstance(result_content, dict):
                    if result_content.get("status") == "error":
                        return True
                # Add more specific checks based on your tool result structure
            except Exception as e:
                self.logger.warning(f"Error checking tool result for errors: {e}")
        return False

    # ==================== 6. User Interaction and Feedback (Interaction Layer) ====================
    # (Copied from original, adapted for modification context)
    def _generate_success_guidance(self, files_modified_count: int) -> str:
        """Generate concise success guidance for continuing iteration"""
        return f"""✅ File modification completed successfully!
📊 **Progress Status:** {files_modified_count} files modified
🎯 **Next Action:** Check if the user's request has been fully addressed.
⚡ **Decision Process:**
1. **If user request is fully addressed:** Reply with "Iteration complete" or similar to end the task.
2. **If MORE changes are needed:** Continue iterating:
   - **Use `read_file` to inspect relevant code if needed**
   - **Use `write_file` to implement the next required change"""

    def _generate_error_guidance(self) -> str:
        """Generate error guidance for handling issues during iteration"""
        return """❌ Error detected during file modification.
🔧 **Action Required:**
1. Review the error details above
2. Fix the identified issue
3. **Check if user request is fully addressed:**
   - **If YES:** Respond "**iteration complete**" to end the conversation
   - **If NO:** Continue iterating:
     - **Use `read_file` to inspect relevant code if needed**
     - **Use `write_file` to implement the next required change
4. Ensure proper error handling in future implementations"""

    def _generate_no_tools_guidance(self, files_modified_count: int) -> str:
        """Generate concise guidance when no tools are called"""
        return f"""⚠️ No tool calls detected in your response.
📊 **Current Progress:** {files_modified_count} files modified
🚨 **Action Required:** Check if user request is addressed NOW:
⚡ **Decision Process:**
1. **If user request is fully addressed:** Reply "Iteration complete" to end
2. **If MORE changes needed:** Use tools to continue:
   - **Use `read_file` to inspect relevant code if needed**
   - **Use `write_file` to implement the next change
🚨 **Critical:** Don't just explain - either declare completion or use tools!"""

    def _compile_user_response(self, tool_results: List[Dict], guidance: str) -> str:
        """Compile tool results and guidance into a single user response (Copied from original)"""
        response_parts = []
        if tool_results:
            response_parts.append("🔧 **Tool Execution Results:**")
            for tool_result in tool_results:
                tool_name = tool_result["tool_name"]
                result_content = tool_result["result"]
                response_parts.append(
                    f'''Tool: {tool_name} Result: {result_content}''')
        if guidance:
            response_parts.append("" + guidance)
        return "".join(response_parts)

    # ==================== 7. Reporting and Output (Output Layer) ====================
    async def _generate_iteration_final_report(
        self,
        iterations: int,
        elapsed_time: float,
        iteration_agent: CodeIterationAgent,
        memory_agent: ConciseMemoryAgent,
        iteration_directory: str 
    ):
        """Generate final report for the iteration process."""
        try:
            iteration_stats = iteration_agent.get_iteration_statistics()
            # Assuming memory_agent has a method to get its stats
            memory_stats = memory_agent.get_memory_statistics(
                iteration_stats["files_modified_count"]
            ) if hasattr(memory_agent, 'get_memory_statistics') else {"message": "No specific memory stats method"}

            if self.mcp_agent:
                history_result = await self.mcp_agent.call_tool(
                    "get_operation_history", {"last_n": 30}
                )
                # history_result is a CallToolResult object, need to extract its content.text
                # Check if history_result has a 'content' attribute and 'text' attribute
                if hasattr(history_result, 'content') and hasattr(history_result.content, 'text'):
                    result_text = history_result.content.text
                elif isinstance(history_result, str):
                    # Fallback if it's already a string
                    result_text = history_result
                else:
                    # Fallback if structure is unexpected
                    self.logger.warning(f"Unexpected history_result structure: {type(history_result)}")
                    result_text = json.dumps({"total_operations": 0, "history": []})

                try:
                    history_data = json.loads(result_text)
                except json.JSONDecodeError as e:
                    self.logger.error(f"Failed to decode history result JSON: {e}")
                    self.logger.debug(f"Raw history result: {history_result}")
                    history_data = {"total_operations": 0, "history": []}
            else:
                history_data = {"total_operations": 0, "history": []}

            write_operations = 0
            files_modified = []
            if "history" in history_data:
                for item in history_data["history"]:
                    if item.get("action") == "write_file": # Or "modify_file" if you track differently
                        write_operations += 1
                        file_path = item.get("details", {}).get("file_path", "unknown")
                        files_modified.append(file_path)

            report = f"""
# Code Iteration Completion Report
## Execution Summary
- Iteration loops completed: {iterations}
- Total elapsed time: {elapsed_time:.2f} seconds
- Files modified: {iteration_stats['total_files_modified']}
- File write operations: {write_operations}"""

            report += """
## Architecture Features
✅ Iterative modification based on user feedback
✅ MCP-compliant tool execution
✅ Production-grade code with comprehensive type hints
✅ Intelligent dependency analysis and file reading
✅ Automated read_file usage for context during iteration
✅ Memory agent integration for context optimization
✅ Separation of concerns between Iteration Agent and Memory Agent
"""

            return report
        except Exception as e:
            self.logger.error(f"Failed to generate final iteration report: {e}")
            return f"Failed to generate final report: {str(e)}"

# --- Example Usage ---
async def main():
    workflow = CodeIterationWorkflow()
    # Example: Modify code in /path/to/project based on user request

    result = await workflow.run_iteration(
        user_intent="In the lora_merger.py file, there is a problem of duplicate import at the end of the file in the test section. Please help me correct it",
        target_directory="/home/user02/deepcode/deepcode-wei/deepcode_lab/papers/14",
        original_code_dir="generate_code", # Copy from this dir
        iteration_dir_name="iteration_bug_fix", # Create this new dir
        initial_code_snapshot="...", # Optional: Pass snapshot from initial implementation (not copied)
        test_report_before="...", # Optional: Pass test report before iteration
        max_iterations=3,
        enable_read_tools=True
    )
    print(result)

if __name__ == "__main__":
    # print(dir(ConciseMemoryAgent))
    # assert False
    asyncio.run(main())