from __future__ import annotations
import os, json, sys
from typing import Optional, Tuple
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Look for .env file in the parent directory (where the main project is)
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✓ Loaded environment variables from {env_path}")
    else:
        # Try current directory
        load_dotenv()
except ImportError:
    print("⚠ python-dotenv not available, using system environment variables only")

try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
    _OPENAI_CLIENT_CLASS = OpenAI
except Exception:
    _OPENAI_AVAILABLE = False
    _OPENAI_CLIENT_CLASS = None

def call_llm_json(prompt: str, model: str = "gpt-3.5-turbo", 
                  temperature: float = 0.3, provider: str = "auto") -> Tuple[Optional[Dict], Optional[str]]:
    """
    Universal LLM JSON call supporting OpenAI and Gemini via LLPM gateway.
    Auto-detects provider based on available environment variables.
    """
    import os
    import json
    import requests
    from typing import Dict, Optional, Tuple
    
    # Auto-detect provider if not specified
    if provider == "auto":
        if os.getenv("LLPM_KEY"):
            provider = "gemini"
        elif os.getenv("OPENAI_API_AGENT_KEY"):
            provider = "openai"
        else:
            return None, "No API keys found for LLM providers"
    
    try:
        if provider == "gemini":
            # Use LLPM gateway for Gemini
            url = "https://llpm.gateway.ai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {os.getenv('LLMP_KEY')}",
                "Content-Type": "application/json"
            }
            data = {
                "model": "gemini-1.5-flash",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature
            }
            response = requests.post(url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            return json.loads(content), None
            
        elif provider == "openai":
            # Use OpenAI directly with new client format
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_AGENT_KEY"))
            
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=500
            )
            content = response.choices[0].message.content
            if content is None:
                return None, "OpenAI returned empty content"
            return json.loads(content), None
            
        else:
            return None, f"Unsupported provider: {provider}"
            
    except json.JSONDecodeError as e:
        return None, f"JSON decode error: {str(e)}"
    except requests.exceptions.RequestException as e:
        return None, f"Connection error."
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"

def call_openai_json(model: str, api_key_env: str, prompt: str, temperature: float = 0.0):
    """
    Call OpenAI API with JSON response format.
    
    Args:
        model: OpenAI model name (e.g., 'gpt-4o-mini')
        api_key_env: Environment variable name containing API key
        prompt: Prompt text for the model
        temperature: Sampling temperature (0.0 for deterministic)
        
    Returns:
        Tuple of (parsed_json_response, error_message)
        If successful: (dict, None)
        If failed: (None, error_string)
    """
    if not _OPENAI_AVAILABLE:
        return None, "openai SDK not available"
    
    # Get API key from environment
    api_key = os.getenv(api_key_env)
    if not api_key:
        # Try alternative key names for backward compatibility
        api_key = os.getenv("OPENAI_API_AGENT_KEY")
        if not api_key:
            return None, f"missing API key in env var {api_key_env} or OPENAI_API_AGENT_KEY"
    
    # Log LLM invocation
    print(f"🤖 GPT LLM invoked for analysis using model: {model}")
    print(f"   📝 Prompt length: {len(prompt)} characters")
    print(f"   🎛️  Temperature: {temperature}")
    
    try:
        if not _OPENAI_CLIENT_CLASS:
            return None, "OpenAI client class not available"
            
        client = _OPENAI_CLIENT_CLASS(api_key=api_key)
        
        # Log the API call
        print(f"   🌐 Making OpenAI API call...")
        
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )

        content = resp.choices[0].message.content
        if content is None:
            error_msg = "received empty response from OpenAI"
            print(f"   ❌ LLM Error: {error_msg}")
            return None, error_msg
            
        parsed_response = json.loads(content)
        
        # Log success
        print(f"   ✅ LLM Analysis complete - received {len(parsed_response)} fields")
        
        return parsed_response, None
    except Exception as e:
        error_msg = str(e)
        print(f"   ❌ LLM Error: {error_msg}")
        return None, error_msg