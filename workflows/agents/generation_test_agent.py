"""
Test Generation Agent for Automated Test Suite Creation

This agent generates comprehensive test suites for implemented codebases,
including unit tests, integration tests, and validation tests.
"""

import json
import os
import logging
from typing import Dict, Any, Optional

from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm import RequestParams

# Import prompts
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from prompts.test_agent_prompt import TEST_GENERATION_AGENT_PROMPT
from utils.llm_utils import get_preferred_llm_class


class TestGenerationAgent:
    """
    Test Generation Agent for creating comprehensive test suites
    
    Responsibilities:
    - Analyze implemented codebase structure
    - Generate unit tests for core components
    - Create integration tests for workflows
    - Generate validation tests for paper results
    - Create test infrastructure (conftest.py, fixtures)
    - Provide test execution documentation
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize Test Generation Agent
        
        Args:
            logger: Logger instance for tracking operations
        """
        self.logger = logger or self._create_default_logger()
        self.mcp_agent = None
        self.llm = None
        
    def _create_default_logger(self) -> logging.Logger:
        """Create default logger if none provided"""
        logger = logging.getLogger(f"{__name__}.TestGenerationAgent")
        logger.setLevel(logging.INFO)
        return logger
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.cleanup()
    
    async def initialize(self):
        """Initialize the MCP agent and attach LLM"""
        try:
            self.logger.info("Initializing Test Generation Agent...")
            
            # Create agent with code-implementation tools (same as code generation phase)
            self.mcp_agent = Agent(
                name="TestGenerationAgent",
                instruction=TEST_GENERATION_AGENT_PROMPT,
                server_names=["code-implementation"],
            )
            
            # Initialize the agent context
            await self.mcp_agent.__aenter__()
            
            # Attach LLM
            self.llm = await self.mcp_agent.attach_llm(get_preferred_llm_class())
            
            self.logger.info("TestGenerationAgent initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize TestGenerationAgent: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup agent resources"""
        if self.mcp_agent:
            try:
                await self.mcp_agent.__aexit__(None, None, None)
                self.logger.info("TestGenerationAgent cleaned up")
            except Exception as e:
                self.logger.warning(f"Error during cleanup: {e}")
    
    async def generate_tests(
        self,
        code_directory: str,
        plan_file_path: str,
        paper_dir: str,
        max_tokens: int = 8192,
        temperature: float = 0.3,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive test suite for the codebase
        
        Args:
            code_directory: Path to the implemented code
            plan_file_path: Path to the implementation plan
            paper_dir: Path to the paper directory
            max_tokens: Maximum tokens for generation
            temperature: Temperature for LLM generation
            
        Returns:
            Dictionary with test generation results
        """
        try:
            self.logger.info(f"📋 Generating tests for code in: {code_directory}")
            self.logger.info(f"📄 Using implementation plan: {plan_file_path}")
            
            # Prepare message for test generation
            message = f"""Generate comprehensive test suite for the implemented codebase.

**Implementation Plan**: {plan_file_path}
**Code Directory**: {code_directory}
**Paper Directory**: {paper_dir}

Please:
1. Analyze the codebase structure in {code_directory}
2. Read the implementation plan from {plan_file_path}
3. Generate a complete test suite including:
   - Unit tests for core components
   - Integration tests for the pipeline
   - Validation tests for paper results
4. Create test structure with conftest.py and test documentation
5. Provide a summary of generated tests

Use the filesystem tools to read code, create directories, and write test files.
"""
            
            # Configure request parameters to use code-implementation server tools
            test_params = RequestParams(
                maxTokens=max_tokens,
                temperature=temperature,
                tool_filter={
                    "code-implementation": {
                        "read_file",
                        "read_multiple_files",
                        "list_directory",
                        "write_file",
                        "create_file",
                        "ensure_workspace_exists",
                    }
                },
            )
            
            self.logger.info("🔄 Starting test generation...")
            
            # Generate tests
            result = await self.llm.generate_str(
                message=message,
                request_params=test_params,
            )
            
            self.logger.info("✅ Test generation completed")
            
            # Parse result
            test_summary = self._parse_result(result, code_directory)
            
            return test_summary
            
        except Exception as e:
            self.logger.error(f"❌ Error during test generation: {e}")
            raise
    
    def _parse_result(self, result: str, code_directory: str) -> Dict[str, Any]:
        """
        Parse test generation result
        
        Args:
            result: Raw result from LLM
            code_directory: Code directory path
            
        Returns:
            Parsed summary dictionary
        """
       

        test_summary = {
            "status": "success",
            "test_directory": os.path.join(code_directory, "tests"),
            "raw_result": result,
            "message": "Test suite generated successfully",
        }
        
        # Try to extract JSON summary if present
        try:
            import re
            json_match = re.search(r'\{[^{}]*"status"[^{}]*\}', result, re.DOTALL)
            if json_match:
                extracted = json.loads(json_match.group())
                test_summary.update(extracted)
        except Exception:
            # Keep default summary if parsing fails
            pass
        
        return test_summary
