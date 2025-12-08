"""
LLM utility functions for DeepCode project.

This module provides common LLM-related utilities to avoid circular imports
and reduce code duplication across the project.
"""

import os
import yaml
from typing import Any, Type, Dict, Tuple, List, Optional
import json
import base64
import requests

# Import LLM classes
from mcp_agent.workflows.llm.augmented_llm_anthropic import AnthropicAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm_google import GoogleAugmentedLLM


def get_preferred_llm_class(config_path: str = "mcp_agent.secrets.yaml") -> Type[Any]:
    """
    Select the LLM class based on user preference and API key availability.

    Priority:
    1. Check mcp_agent.config.yaml for llm_provider preference
    2. Verify the preferred provider has API key
    3. Fallback to first available provider

    Args:
        config_path: Path to the secrets YAML configuration file

    Returns:
        class: The preferred LLM class
    """
    try:
        # Read API keys from secrets file
        if not os.path.exists(config_path):
            print(f"🤖 Config file {config_path} not found, using OpenAIAugmentedLLM")
            return OpenAIAugmentedLLM

        with open(config_path, "r", encoding="utf-8") as f:
            secrets = yaml.safe_load(f)

        # Get API keys
        anthropic_key = secrets.get("anthropic", {}).get("api_key", "").strip()
        google_key = secrets.get("google", {}).get("api_key", "").strip()
        openai_key = secrets.get("openai", {}).get("api_key", "").strip()

        # Read user preference from main config
        main_config_path = "mcp_agent.config.yaml"
        preferred_provider = None
        if os.path.exists(main_config_path):
            with open(main_config_path, "r", encoding="utf-8") as f:
                main_config = yaml.safe_load(f)
                preferred_provider = main_config.get("llm_provider", "").strip().lower()

        # Map of providers to their classes and keys
        provider_map = {
            "anthropic": (
                AnthropicAugmentedLLM,
                anthropic_key,
                "AnthropicAugmentedLLM",
            ),
            "google": (GoogleAugmentedLLM, google_key, "GoogleAugmentedLLM"),
            "openai": (OpenAIAugmentedLLM, openai_key, "OpenAIAugmentedLLM"),
        }

        # Try user's preferred provider first
        if preferred_provider and preferred_provider in provider_map:
            llm_class, api_key, class_name = provider_map[preferred_provider]
            if api_key:
                print(f"🤖 Using {class_name} (user preference: {preferred_provider})")
                return llm_class
            else:
                print(
                    f"⚠️ Preferred provider '{preferred_provider}' has no API key, checking alternatives..."
                )

        # Fallback: try providers in order of availability
        for provider, (llm_class, api_key, class_name) in provider_map.items():
            if api_key:
                print(f"🤖 Using {class_name} ({provider} API key found)")
                return llm_class

        # No API keys found
        print("⚠️ No API keys configured, falling back to OpenAIAugmentedLLM")
        return OpenAIAugmentedLLM

    except Exception as e:
        print(f"🤖 Error reading config file {config_path}: {e}")
        print("🤖 Falling back to OpenAIAugmentedLLM")
        return OpenAIAugmentedLLM


def get_token_limits(config_path: str = "mcp_agent.config.yaml") -> Tuple[int, int]:
    """
    Get token limits from configuration.

    Args:
        config_path: Path to the main configuration file

    Returns:
        tuple: (base_max_tokens, retry_max_tokens)
    """
    # Default values that work with qwen/qwen-max (32768 total context)
    default_base = 20000
    default_retry = 15000

    try:
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            openai_config = config.get("openai", {})
            base_tokens = openai_config.get("base_max_tokens", default_base)
            retry_tokens = openai_config.get("retry_max_tokens", default_retry)

            print(
                f"⚙️ Token limits from config: base={base_tokens}, retry={retry_tokens}"
            )
            return base_tokens, retry_tokens
        else:
            print(
                f"⚠️ Config file {config_path} not found, using defaults: base={default_base}, retry={default_retry}"
            )
            return default_base, default_retry
    except Exception as e:
        print(f"⚠️ Error reading token config from {config_path}: {e}")
        print(
            f"🔧 Falling back to default token limits: base={default_base}, retry={default_retry}"
        )
        return default_base, default_retry


def get_default_models(config_path: str = "mcp_agent.config.yaml"):
    """
    Get default models from configuration file.

    Args:
        config_path: Path to the configuration file

    Returns:
        dict: Dictionary with 'anthropic', 'openai', and 'google' default models
    """
    try:
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            # Handle null values in config sections
            anthropic_config = config.get("anthropic") or {}
            openai_config = config.get("openai") or {}
            google_config = config.get("google") or {}

            anthropic_model = anthropic_config.get(
                "default_model", "claude-sonnet-4-20250514"
            )
            openai_model = openai_config.get("default_model", "o3-mini")
            google_model = google_config.get("default_model", "gemini-2.0-flash")

            return {
                "anthropic": anthropic_model,
                "openai": openai_model,
                "google": google_model,
            }
        else:
            print(f"Config file {config_path} not found, using default models")
            return {
                "anthropic": "claude-sonnet-4-20250514",
                "openai": "o3-mini",
                "google": "gemini-2.0-flash",
            }

    except Exception as e:
        print(f"❌Error reading config file {config_path}: {e}")
        return {
            "anthropic": "claude-sonnet-4-20250514",
            "openai": "o3-mini",
            "google": "gemini-2.0-flash",
        }


def get_document_segmentation_config(
    config_path: str = "mcp_agent.config.yaml",
) -> Dict[str, Any]:
    """
    Get document segmentation configuration from config file.

    Args:
        config_path: Path to the main configuration file

    Returns:
        Dict containing segmentation configuration with default values
    """
    try:
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            # Get document segmentation config with defaults
            seg_config = config.get("document_segmentation", {})
            return {
                "enabled": seg_config.get("enabled", True),
                "size_threshold_chars": seg_config.get("size_threshold_chars", 50000),
            }
        else:
            print(
                f"📄 Config file {config_path} not found, using default segmentation settings"
            )
            return {"enabled": True, "size_threshold_chars": 50000}

    except Exception as e:
        print(f"📄 Error reading segmentation config from {config_path}: {e}")
        print("📄 Using default segmentation settings")
        return {"enabled": True, "size_threshold_chars": 50000}


def ensure_ark_api_key_env(secrets_path: str = "mcp_agent.secrets.yaml") -> None:
    try:
        if os.path.exists(secrets_path):
            with open(secrets_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            openai_cfg = cfg.get("openai") or {}
            key = str(openai_cfg.get("api_key", ""))
            if key:
                os.environ["ARK_API_KEY"] = key
    except Exception:
        pass


def should_use_document_segmentation(
    document_content: str, config_path: str = "mcp_agent.config.yaml"
) -> Tuple[bool, str]:
    """
    Determine whether to use document segmentation based on configuration and document size.

    Args:
        document_content: The content of the document to analyze
        config_path: Path to the configuration file

    Returns:
        Tuple of (should_segment, reason) where:
        - should_segment: Boolean indicating whether to use segmentation
        - reason: String explaining the decision
    """
    seg_config = get_document_segmentation_config(config_path)

    if not seg_config["enabled"]:
        return False, "Document segmentation disabled in configuration"

    doc_size = len(document_content)
    threshold = seg_config["size_threshold_chars"]

    if doc_size > threshold:
        return (
            True,
            f"Document size ({doc_size:,} chars) exceeds threshold ({threshold:,} chars)",
        )
    else:
        return (
            False,
            f"Document size ({doc_size:,} chars) below threshold ({threshold:,} chars)",
        )


def get_adaptive_agent_config(
    use_segmentation: bool, search_server_names: list = None
) -> Dict[str, list]:
    """
    Get adaptive agent configuration based on whether to use document segmentation.

    Args:
        use_segmentation: Whether to include document-segmentation server
        search_server_names: Base search server names (from get_search_server_names)

    Returns:
        Dict containing server configurations for different agents
    """
    if search_server_names is None:
        search_server_names = []

    # Base configuration
    config = {
        "concept_analysis": [],
        "algorithm_analysis": search_server_names.copy(),
        "code_planner": search_server_names.copy(),
    }

    # Add document-segmentation server if needed
    if use_segmentation:
        config["concept_analysis"] = ["document-segmentation"]
        if "document-segmentation" not in config["algorithm_analysis"]:
            config["algorithm_analysis"].append("document-segmentation")
        if "document-segmentation" not in config["code_planner"]:
            config["code_planner"].append("document-segmentation")
    else:
        config["concept_analysis"] = ["filesystem"]
        if "filesystem" not in config["algorithm_analysis"]:
            config["algorithm_analysis"].append("filesystem")
        if "filesystem" not in config["code_planner"]:
            config["code_planner"].append("filesystem")

    return config


def get_adaptive_prompts(use_segmentation: bool) -> Dict[str, str]:
    """
    Get appropriate prompt versions based on segmentation usage.

    Args:
        use_segmentation: Whether to use segmented reading prompts

    Returns:
        Dict containing prompt configurations
    """
    # Import here to avoid circular imports
    from prompts.code_prompts import (
        PAPER_CONCEPT_ANALYSIS_PROMPT,
        PAPER_ALGORITHM_ANALYSIS_PROMPT,
        CODE_PLANNING_PROMPT,
        PAPER_CONCEPT_ANALYSIS_PROMPT_TRADITIONAL,
        PAPER_ALGORITHM_ANALYSIS_PROMPT_TRADITIONAL,
        CODE_PLANNING_PROMPT_TRADITIONAL,
    )

    if use_segmentation:
        return {
            "concept_analysis": PAPER_CONCEPT_ANALYSIS_PROMPT,
            "algorithm_analysis": PAPER_ALGORITHM_ANALYSIS_PROMPT,
            "code_planning": CODE_PLANNING_PROMPT,
        }
    else:
        return {
            "concept_analysis": PAPER_CONCEPT_ANALYSIS_PROMPT_TRADITIONAL,
            "algorithm_analysis": PAPER_ALGORITHM_ANALYSIS_PROMPT_TRADITIONAL,
            "code_planning": CODE_PLANNING_PROMPT_TRADITIONAL,
        }


def _find_paper_markdown(paper_dir: str) -> Tuple[str, str]:
    md_path = ""
    md_content = ""
    try:
        for filename in os.listdir(paper_dir):
            if filename.endswith(".md"):
                md_path = os.path.join(paper_dir, filename)
                with open(md_path, "r", encoding="utf-8") as f:
                    md_content = f.read()
                break
    except Exception:
        pass
    return md_path, md_content


def _extract_md_images(md_content: str, paper_dir: str, limit: int = 20) -> list:
    import re
    paths = []
    md_pattern = r"!\[[^\]]*\]\(([^)]+)\)"
    html_pattern = r"<img[^>]*src=[\"']([^\"']+)[\"'][^>]*>"
    candidates = []
    candidates.extend(re.findall(md_pattern, md_content))
    candidates.extend(re.findall(html_pattern, md_content))
    for p in candidates:
        p = p.strip()
        if not p:
            continue
        if p.startswith("http://") or p.startswith("https://"):
            continue
        if p.startswith("data:image"):
            continue
        if os.path.isabs(p):
            abs_path = p
        else:
            abs_path = os.path.join(paper_dir, p)
        if os.path.exists(abs_path):
            paths.append(abs_path)
        if len(paths) >= limit:
            break
    return paths


def _encode_file_to_data_url(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower().replace(".", "")
    if ext == "jpg":
        ext = "jpeg"
    with open(file_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/{ext};base64,{b64}"


def collect_paper_images_as_data_urls(paper_dir: str, max_images: int = 20) -> Tuple[str, list]:
    md_path, md_content = _find_paper_markdown(paper_dir)
    if not md_path or not md_content:
        return "", []
    image_files = _extract_md_images(md_content, os.path.dirname(md_path), max_images)
    data_urls = []
    for p in image_files:
        try:
            data_urls.append(_encode_file_to_data_url(p))
        except Exception:
            continue
    return md_content, data_urls


def collect_paper_image_file_paths(
    paper_dir: str, max_images: int = 20
) -> Tuple[str, List[str]]:
    md_path, md_content = _find_paper_markdown(paper_dir)
    if not md_path or not md_content:
        images_dir = os.path.join(paper_dir, "images")
        files: List[str] = []
        if os.path.isdir(images_dir):
            allowed_ext = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}
            files = [
                os.path.join(images_dir, f)
                for f in os.listdir(images_dir)
                if os.path.splitext(f)[1].lower() in allowed_ext
            ]
        else:
            allowed_ext = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}
            files = [
                os.path.join(paper_dir, f)
                for f in os.listdir(paper_dir)
                if os.path.splitext(f)[1].lower() in allowed_ext
            ]
        files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return "", files[:max_images]
    images_from_md = _extract_md_images(md_content, os.path.dirname(md_path), max_images)
    image_files = list(images_from_md)
    if len(image_files) < max_images:
        allowed_ext = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}
        same_dir_files = [
            os.path.join(os.path.dirname(md_path), f)
            for f in os.listdir(os.path.dirname(md_path))
            if os.path.splitext(f)[1].lower() in allowed_ext
        ]
        for p in same_dir_files:
            if p not in image_files:
                image_files.append(p)
                if len(image_files) >= max_images:
                    break
    if len(image_files) < max_images:
        images_dir = os.path.join(paper_dir, "images")
        if os.path.isdir(images_dir):
            allowed_ext = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}
            extra = [
                os.path.join(images_dir, f)
                for f in os.listdir(images_dir)
                if os.path.splitext(f)[1].lower() in allowed_ext
            ]
            for p in extra:
                if p not in image_files:
                    image_files.append(p)
                    if len(image_files) >= max_images:
                        break
    return md_content, image_files


def create_openai_client_from_secrets(config_path: str = "mcp_agent.secrets.yaml"):
    try:
        from openai import OpenAI
    except Exception:
        return None
    try:
        if not os.path.exists(config_path):
            return None
        with open(config_path, "r", encoding="utf-8") as f:
            secrets = yaml.safe_load(f)
        base_url = secrets.get("openai", {}).get("base_url")
        ensure_ark_api_key_env(config_path)
        api_key_env = os.environ.get("ARK_API_KEY")
        if not api_key_env or not base_url:
            return None
        client = OpenAI(api_key=os.environ.get("ARK_API_KEY"), base_url=base_url)
        return client
    except Exception:
        return None


def create_ark_client_from_secrets(config_path: str = "mcp_agent.secrets.yaml"):
    try:
        from volcenginesdkarkruntime import AsyncArk
    except Exception:
        return None
    try:
        if not os.path.exists(config_path):
            return None
        with open(config_path, "r", encoding="utf-8") as f:
            secrets = yaml.safe_load(f)
        base_url = secrets.get("openai", {}).get("base_url")
        ensure_ark_api_key_env(config_path)
        api_key_env = os.environ.get("ARK_API_KEY")
        if not api_key_env or not base_url:
            return None
        client = AsyncArk(base_url=base_url, api_key=os.environ.get("ARK_API_KEY"))
        return client
    except Exception:
        return None


def _http_responses_create(
    base_url: str,
    api_key: str,
    model_name: str,
    input_blocks: List[Dict[str, Any]],
    max_tokens: int,
    reasoning: Optional[Dict[str, Any]] = None,
) -> str:
    url = base_url.rstrip("/") + "/responses"
    api_key_env = os.environ.get("ARK_API_KEY") or api_key
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key_env}",
    }
    payload = {
        "model": model_name,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": input_blocks,
            }
        ],
        "max_tokens": max_tokens,
    }
    if reasoning:
        payload["reasoning"] = reasoning
    resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
    if resp.status_code >= 200 and resp.status_code < 300:
        data = resp.json()
        if "output_text" in data and isinstance(data["output_text"], str):
            return data["output_text"]
        if "output" in data and isinstance(data["output"], list):
            parts = []
            for item in data["output"]:
                if "content" in item and isinstance(item["content"], list):
                    for c in item["content"]:
                        if c.get("type") == "output_text":
                            parts.append(c.get("text", ""))
            return "\n".join(parts).strip()
        return json.dumps(data, ensure_ascii=False)
    return ""

def _http_upload_file(base_url: str, api_key: str, file_path: str, purpose: str = "user_data") -> Optional[str]:
    url = base_url.rstrip("/") + "/files"
    api_key_env = os.environ.get("ARK_API_KEY") or api_key
    headers = {
        "Authorization": f"Bearer {api_key_env}",
    }
    try:
        with open(file_path, "rb") as f:
            files = {
                "file": (os.path.basename(file_path), f),
            }
            data = {"purpose": purpose}
            resp = requests.post(url, headers=headers, files=files, data=data, timeout=120)
        if resp.status_code >= 200 and resp.status_code < 300:
            j = resp.json()
            fid = j.get("id") or j.get("file_id")
            return fid
    except Exception:
        pass
    return None

def _http_chat_completions(
    base_url: str,
    api_key: str,
    model_name: str,
    messages: List[Dict[str, Any]],
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('ARK_API_KEY') or api_key}",
    }
    payload: Dict[str, Any] = {
        "model": model_name,
        "messages": messages,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if temperature is not None:
        payload["temperature"] = temperature
    resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
    if resp.status_code >= 200 and resp.status_code < 300:
        data = resp.json()
        try:
            choices = data.get("choices") or []
            if choices:
                msg = choices[0].get("message") or {}
                content = msg.get("content")
                if isinstance(content, list):
                    parts = []
                    for c in content:
                        if c.get("type") in ("text", "output_text"):
                            parts.append(c.get("text", ""))
                    return "\n".join(parts).strip()
                if isinstance(content, str):
                    return content
        except Exception:
            pass
        return json.dumps(data, ensure_ascii=False)
    return ""


async def generate_multimodal_plan(
    paper_dir: str,
    task_prompt: str,
    model_name: str,
    max_tokens: int = 12000,
    temperature: float = 0.2,
    config_path: str = "mcp_agent.config.yaml",
) -> str:
    ensure_ark_api_key_env("mcp_agent.secrets.yaml")
    md_text, file_paths = collect_paper_image_file_paths(paper_dir)
    if not md_text and not file_paths:
        return ""

    reasoning = None
    try:
        if os.path.exists("mcp_agent.config.yaml"):
            with open("mcp_agent.config.yaml", "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
                openai_cfg = cfg.get("openai") or {}
                eff = openai_cfg.get("reasoning_effort")
                if isinstance(eff, str) and eff:
                    reasoning = {"effort": eff}
    except Exception:
        reasoning = None

    base_url = None
    try:
        with open("mcp_agent.secrets.yaml", "r", encoding="utf-8") as f:
            secrets = yaml.safe_load(f)
        base_url = secrets.get("openai", {}).get("base_url")
        # Respect configured default model if provided
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            openai_cfg = cfg.get("openai") or {}
            configured_model = openai_cfg.get("default_model")
            if isinstance(configured_model, str) and configured_model:
                model_name = configured_model
    except Exception:
        base_url = None

    api_key_env = os.environ.get("ARK_API_KEY")
    if api_key_env and base_url:
        file_ids: List[str] = []
        for p in file_paths:
            fid = _http_upload_file(base_url=base_url, api_key=api_key_env, file_path=p)
            if fid:
                file_ids.append(fid)

        if file_ids:
            blocks = []
            for fid in file_ids:
                blocks.append({"type": "input_image", "file_id": fid})
            if md_text:
                blocks.append({"type": "input_text", "text": task_prompt + "\n\n" + md_text})
            else:
                blocks.append({"type": "input_text", "text": task_prompt + "\n\n[Images only; no markdown text detected]"})
            out = _http_responses_create(
                base_url=base_url,
                api_key=api_key_env,
                model_name=model_name,
                input_blocks=blocks,
                max_tokens=max_tokens,
                reasoning=reasoning,
            )
            if out:
                return out

        try:
            msgs: List[Dict[str, Any]] = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": task_prompt + "\n\n" + md_text},
                    ],
                }
            ]
            for p in file_paths:
                try:
                    with open(p, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode("ascii")
                    ext = os.path.splitext(p)[1].lower().strip(".")
                    if ext == "jpg":
                        ext = "jpeg"
                    data_url = f"data:image/{ext};base64,{b64}"
                    msgs[0]["content"].insert(0, {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}})
                except Exception:
                    pass
            out2 = _http_chat_completions(
                base_url=base_url,
                api_key=api_key_env,
                model_name=model_name,
                messages=msgs,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            if out2:
                return out2
        except Exception:
            pass

        # Final attempt via HTTP /responses using base64 data URLs directly
        try:
            blocks_b64: List[Dict[str, Any]] = []
            for p in file_paths:
                try:
                    with open(p, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode("ascii")
                    ext = os.path.splitext(p)[1].lower().strip(".")
                    if ext == "jpg":
                        ext = "jpeg"
                    data_url = f"data:image/{ext};base64,{b64}"
                    blocks_b64.append({"type": "input_image", "image_url": data_url})
                except Exception:
                    blocks_b64.append({"type": "input_text", "text": f"[image missing: {p}]"})
            if md_text:
                blocks_b64.append({"type": "input_text", "text": task_prompt + "\n\n" + md_text})
            else:
                blocks_b64.append({"type": "input_text", "text": task_prompt + "\n\n[Images only; no markdown text detected]"})
            out3 = _http_responses_create(
                base_url=base_url,
                api_key=api_key_env,
                model_name=model_name,
                input_blocks=blocks_b64,
                max_tokens=max_tokens,
                reasoning=reasoning,
            )
            if out3:
                return out3
        except Exception:
            pass

    ark_client = create_ark_client_from_secrets()
    if ark_client is not None and file_paths:
        try:
            input_blocks_file = []
            for p in file_paths:
                try:
                    with open(p, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode("ascii")
                    ext = os.path.splitext(p)[1].lower()
                    mime = {
                        ".png": "image/png",
                        ".jpg": "image/jpeg",
                        ".jpeg": "image/jpeg",
                        ".gif": "image/gif",
                        ".webp": "image/webp",
                        ".svg": "image/svg+xml",
                    }.get(ext, "image/png")
                    data_url = f"data:{mime};base64,{b64}"
                    input_blocks_file.append({"type": "input_image", "image_url": data_url})
                except Exception:
                    input_blocks_file.append({"type": "input_text", "text": f"[image] {p}"})
            if md_text:
                input_blocks_file.append({"type": "input_text", "text": task_prompt + "\n\n" + md_text})
            else:
                input_blocks_file.append({"type": "input_text", "text": task_prompt + "\n\n[Images only; no markdown text detected]"})
            kwargs = {
                "model": model_name,
                "input": [{"role": "user", "content": input_blocks_file}],
                "max_tokens": max_tokens,
            }
            if reasoning:
                kwargs["reasoning"] = reasoning
            response = await ark_client.responses.create(**kwargs)
            content = getattr(response, "output_text", None)
            if content:
                return content
            if hasattr(response, "output") and isinstance(response.output, list):
                parts = []
                for item in response.output:
                    if hasattr(item, "content") and isinstance(item.content, list):
                        for c in item.content:
                            if c.get("type") == "output_text":
                                parts.append(c.get("text", ""))
                out = "\n".join(parts).strip()
                if out:
                    return out
            return str(response)
        except Exception:
            pass

    md_text2, data_urls = collect_paper_images_as_data_urls(paper_dir)
    if not md_text2 and not data_urls:
        return ""
    openai_client = create_openai_client_from_secrets()
    if openai_client is not None:
        input_blocks_b64 = []
        for url in data_urls:
            input_blocks_b64.append({"type": "input_image", "image_url": url})
        if md_text2:
            input_blocks_b64.append({"type": "input_text", "text": task_prompt + "\n\n" + md_text2})
        else:
            input_blocks_b64.append({"type": "input_text", "text": task_prompt + "\n\n[Images only; no markdown text detected]"})
        try:
            kwargs = {
                "model": model_name,
                "input": [{"role": "user", "content": input_blocks_b64}],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if reasoning:
                kwargs["reasoning"] = reasoning
            response = openai_client.responses.create(**kwargs)
            content = getattr(response, "output_text", None)
            if content:
                return content
            if hasattr(response, "output") and isinstance(response.output, list):
                parts = []
                for item in response.output:
                    if hasattr(item, "content") and isinstance(item.content, list):
                        for c in item.content:
                            if c.get("type") == "output_text":
                                parts.append(c.get("text", ""))
                return "\n".join(parts).strip()
            return str(response)
        except Exception:
            return ""
    return ""

def is_ark_openai(secrets_path: str = "mcp_agent.secrets.yaml") -> Tuple[bool, str]:
    try:
        if os.path.exists(secrets_path):
            with open(secrets_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            openai_cfg = cfg.get("openai") or {}
            base_url = str(openai_cfg.get("base_url", ""))
            return ("ark.cn-beijing.volces.com" in base_url), base_url
    except Exception:
        pass
    return False, ""

def collect_media_assets(
    paper_dir: str, max_images: int = 16, max_tables: int = 8
) -> Dict[str, Any]:
    images: List[str] = []
    tables: List[str] = []
    images_dir = os.path.join(paper_dir, "images")
    if os.path.isdir(images_dir):
        allowed_ext = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}
        files = [
            os.path.join(images_dir, f)
            for f in os.listdir(images_dir)
            if os.path.splitext(f)[1].lower() in allowed_ext
        ]
        files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        images = files[:max_images]

    try:
        from utils.file_processor import FileProcessor
        md_path = FileProcessor.find_markdown_file(paper_dir)
        if md_path and os.path.exists(md_path):
            with open(md_path, "r", encoding="utf-8") as f:
                content = f.read()
            segments: List[str] = []
            lines = content.splitlines()
            i = 0
            while i < len(lines):
                line = lines[i]
                if "|" in line and line.strip().startswith("|"):
                    block = [line]
                    j = i + 1
                    while j < len(lines) and "|" in lines[j] and lines[j].strip().startswith("|"):
                        block.append(lines[j])
                        j += 1
                    segments.append("\n".join(block))
                    i = j
                    continue
                if "<table" in line:
                    block = [line]
                    j = i + 1
                    while j < len(lines):
                        block.append(lines[j])
                        if "</table>" in lines[j]:
                            break
                        j += 1
                    segments.append("\n".join(block))
                    i = j + 1
                    continue
                i += 1
            for seg in segments[:max_tables]:
                if len(seg) > 4000:
                    tables.append(seg[:4000])
                else:
                    tables.append(seg)
    except Exception:
        pass

    return {"images": images, "tables": tables}

def compose_media_section(media: Dict[str, Any]) -> str:
    parts: List[str] = []
    imgs: List[str] = media.get("images") or []
    tbls: List[str] = media.get("tables") or []
    if imgs:
        parts.append("IMAGES:")
        for p in imgs:
            parts.append(f"- file://{p}")
    if tbls:
        parts.append("TABLES:")
        for t in tbls:
            parts.append(t)
            parts.append("---")
    return "\n".join(parts)

def build_openai_multimodal_user_content(
    text: str, images: List[str], tables: List[str]
) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = []
    if text:
        content.append({"type": "text", "text": text})

    def _guess_mime(path: str) -> str:
        ext = os.path.splitext(path)[1].lower()
        return {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".svg": "image/svg+xml",
        }.get(ext, "image/png")

    for img in images or []:
        try:
            with open(img, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("ascii")
            mime = _guess_mime(img)
            data_url = f"data:{mime};base64,{b64}"
            content.append({
                "type": "image_url",
                "image_url": {"url": data_url},
            })
        except Exception:
            content.append({"type": "text", "text": f"[image] {img}"})

    for tbl in tables or []:
        if tbl:
            content.append({"type": "text", "text": tbl})

    return content
