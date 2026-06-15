"""
LLM Interface Module for HSeg

Provides a unified interface for interacting with public LLM backends
(HuggingFace, Gemini, OpenAI, Local Transformers, Mock) with robust JSON extraction.
Confidential custom endpoints are removed.
"""

import json
import os
import re
import logging
import time
import uuid
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class LLMInterface(ABC):
    """Abstract base class for LLM interfaces."""
    
    def __init__(self, model_name: str, temperature: float = 0.1, max_tokens: int = 2048, **kwargs):
        """
        Initialize LLM interface.
        
        Args:
            model_name: Model identifier
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters (ignored by base class)
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        logger.info(f"Initialized LLM interface: {model_name}")
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate response from LLM.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        pass
    
    def extract_json(self, response: str) -> List[Dict[str, Any]]:
        """
        Extract JSON array from LLM response.
        
        Handles cases where model adds explanation before/after JSON.
        Tries multiple strategies to extract valid JSON.
        
        Args:
            response: LLM response text
            
        Returns:
            Parsed JSON array
            
        Raises:
            ValueError: If no valid JSON found
        """
        # Strategy 1: Try to parse entire response as-is
        try:
            parsed = json.loads(response)
            if isinstance(parsed, list):
                logger.debug(f"Parsed entire response as JSON with {len(parsed)} items")
                return parsed
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Look for JSON array within code blocks (markdown)
        code_block_match = re.search(r'```(?:json)?\s*\n(.*?)\n\s*```', response, re.DOTALL)
        if code_block_match:
            try:
                parsed = json.loads(code_block_match.group(1))
                if isinstance(parsed, list):
                    logger.debug(f"Extracted JSON from code block with {len(parsed)} items")
                    return parsed
            except json.JSONDecodeError as e:
                logger.debug(f"Code block JSON parse failed: {e}")
        
        # Strategy 3: Find first [ and last ] and try to parse (handles wrapped JSON)
        first_bracket = response.find('[')
        last_bracket = response.rfind(']')
        
        if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
            json_candidate = response[first_bracket:last_bracket + 1]
            try:
                parsed = json.loads(json_candidate)
                if isinstance(parsed, list):
                    logger.debug(f"Extracted JSON from bracket positions with {len(parsed)} items")
                    return parsed
            except json.JSONDecodeError:
                logger.debug(f"Failed to parse JSON between brackets")
        
        # Strategy 4: Look for JSON-like lines (one object per line)
        json_lines = []
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('{') and line.endswith('}'):
                try:
                    obj = json.loads(line)
                    json_lines.append(obj)
                except json.JSONDecodeError:
                    continue
        
        if json_lines:
            logger.debug(f"Extracted {len(json_lines)} JSON objects from lines")
            return json_lines
        
        # Strategy 5: Strip code block markers and try again
        if '```' in response:
            cleaned = re.sub(r'```(?:json)?\s*\n?', '', response)
            cleaned = cleaned.strip()
            try:
                parsed = json.loads(cleaned)
                if isinstance(parsed, list):
                    logger.debug(f"Extracted JSON after stripping code blocks with {len(parsed)} items")
                    return parsed
            except json.JSONDecodeError:
                pass
        
        # If all strategies fail, show what we tried
        preview = response[:300].replace('\n', ' ') if response else "(empty)"
        logger.error(f"Failed to extract JSON from response. Preview: {preview}...")
        raise ValueError(f"No valid JSON array found in response")
    
    def generate_with_retry(self, prompt: str, max_retries: int = 5, **kwargs) -> str:
        """
        Generate with automatic retry on failure with exponential backoff.
        
        Args:
            prompt: Input prompt
            max_retries: Maximum number of retry attempts
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
            
        Raises:
            Exception: If all retries fail
        """
        base_delay = kwargs.pop('base_delay', 2.0)  # Base delay in seconds
        
        for attempt in range(max_retries):
            try:
                return self.generate(prompt, **kwargs)
            except Exception as e:
                error_str = str(e)
                
                # Check if it's a rate limit error
                is_rate_limit = '429' in error_str or 'RESOURCE_EXHAUSTED' in error_str or 'rate limit' in error_str.lower()
                
                if attempt == max_retries - 1:
                    raise
                
                # Calculate delay with exponential backoff
                if is_rate_limit:
                    delay = min(base_delay * (5 ** attempt), 60.0)
                    logger.warning(f"Rate limit hit (attempt {attempt + 1}/{max_retries}). "
                                 f"Waiting {delay:.1f}s before retry...")
                else:
                    delay = base_delay * (2 ** attempt)  # 2s, 4s, 8s, 16s, 32s
                    logger.warning(f"Generation attempt {attempt + 1}/{max_retries} failed: {e}. "
                                 f"Retrying in {delay:.1f}s...")
                
                time.sleep(delay)
        
        raise Exception("All retry attempts failed")


class HuggingFaceInterface(LLMInterface):
    """Interface for HuggingFace Inference API."""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None, **kwargs):
        """
        Initialize HuggingFace interface.
        
        Args:
            model_name: HF model ID (e.g., "meta-llama/Llama-3.1-70B-Instruct")
            api_key: HuggingFace API token
            **kwargs: Additional parameters
        """
        super().__init__(model_name, **kwargs)
        self.api_key = api_key
        
        try:
            from huggingface_hub import InferenceClient
            self.client = InferenceClient(token=api_key)
            logger.info(f"Initialized HuggingFace client for {model_name}")
        except ImportError:
            logger.error("huggingface_hub not installed. Install with: pip install huggingface_hub")
            raise
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate using HuggingFace Inference API."""
        temperature = kwargs.get('temperature', self.temperature)
        max_tokens = kwargs.get('max_tokens', self.max_tokens)
        
        try:
            response = self.client.text_generation(
                prompt,
                model=self.model_name,
                temperature=temperature,
                max_new_tokens=max_tokens,
                return_full_text=False
            )
            return response
        except Exception as e:
            logger.error(f"HuggingFace API error: {e}")
            raise


class OpenAIInterface(LLMInterface):
    """Interface for standard OpenAI API."""
    
    def __init__(self, model_name: str = "gpt-4o", api_key: Optional[str] = None, **kwargs):
        """
        Initialize OpenAI interface.
        
        Args:
            model_name: OpenAI model name (e.g., "gpt-4o", "gpt-4")
            api_key: OpenAI API key (or set OPENAI_API_KEY env variable)
            **kwargs: Additional parameters
        """
        super().__init__(model_name, **kwargs)
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
            logger.info(f"Initialized OpenAI client for {model_name}")
        except ImportError:
            logger.error("openai not installed. Install with: pip install openai")
            raise
            
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response using OpenAI chat completion API."""
        temperature = kwargs.get('temperature', self.temperature)
        max_tokens = kwargs.get('max_tokens', self.max_tokens)
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise


class GeminiInterface(LLMInterface):
    """Interface for Google Gemini API with rate limiting."""
    
    def __init__(self, model_name: str = "gemini-2.5-flash-lite", api_key: Optional[str] = None, **kwargs):
        """
        Initialize Gemini interface.
        
        Args:
            model_name: Gemini model name
            api_key: Google API key (or set GEMINI_API_KEY env variable)
            **kwargs: Additional parameters including:
                - rpm_limit: Requests per minute limit (default: 1000)
        """
        super().__init__(model_name, **kwargs)
        self.rpm_limit = kwargs.get('rpm_limit', 1000)
        self.min_delay = 60.0 / self.rpm_limit
        self.last_request_time = 0
        self.thinking_level = kwargs.get('thinking_level', None)
        
        self.api_key = api_key or os.environ.get('GEMINI_API_KEY')
        if not self.api_key:
            logger.error("Gemini API key not provided.")
            raise ValueError("Gemini API key required")
        
        try:
            from google import genai
            self.client = genai.Client(api_key=self.api_key)
            logger.info(f"Initialized Gemini client for {model_name}")
        except ImportError:
            logger.error("google-genai not installed. Install with: pip install google-genai")
            raise
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate using Google Gemini API with rate limiting."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.min_delay:
            sleep_time = self.min_delay - time_since_last
            time.sleep(sleep_time)

        temperature = kwargs.get('temperature', self.temperature)
        max_tokens = kwargs.get('max_tokens', self.max_tokens)
        json_schema = kwargs.get('json_schema', None)
        thinking_level = kwargs.get('thinking_level', self.thinking_level)

        gen_config: Dict[str, Any] = {
            'temperature': temperature,
            'max_output_tokens': max_tokens,
        }

        if json_schema is not None:
            gen_config['response_mime_type'] = 'application/json'
            gen_config['response_json_schema'] = json_schema

        if thinking_level is not None:
            try:
                from google.genai import types as _genai_types
                gen_config['thinking_config'] = _genai_types.ThinkingConfig(
                    thinking_level=thinking_level
                )
            except (ImportError, AttributeError) as e:
                logger.warning(f"Could not set thinking_level ({e}); proceeding without it")

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=gen_config,
            )
            self.last_request_time = time.time()

            result_text = ""
            if getattr(response, 'candidates', None) and response.candidates:
                candidate = response.candidates[0]
                if getattr(candidate, 'content', None) and getattr(candidate.content, 'parts', None):
                    text_parts = []
                    for part in candidate.content.parts:
                        if getattr(part, 'text', None):
                            text_parts.append(part.text)
                    result_text = "".join(text_parts)

            if not result_text:
                genai_logger = logging.getLogger('google_genai.types')
                original_level = genai_logger.level
                genai_logger.setLevel(logging.ERROR)
                try:
                    if getattr(response, 'text', None):
                        result_text = response.text
                finally:
                    genai_logger.setLevel(original_level)

            if not result_text:
                num_candidates = len(response.candidates) if hasattr(response, 'candidates') and response.candidates else 0
                raise ValueError(f"Model returned empty response. Candidates: {num_candidates}")

            return result_text
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise


class LocalTransformersInterface(LLMInterface):
    """Interface for running models locally using HuggingFace Transformers."""

    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, **kwargs)

        if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            logger.info("Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to prevent OOM")

        try:
            import gc
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoConfig

            self._gc = gc
            self._torch = torch
            
            logger.info(f"Initializing Local Transformers for {model_name}")

            hf_token = os.environ.get("HF_TOKEN")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
            
            load_in_4bit = kwargs.get('load_in_4bit', False)
            load_in_8bit = kwargs.get('load_in_8bit', False)
            use_multi_gpu = kwargs.get('use_multi_gpu', False)
            
            model_kwargs = {
                "trust_remote_code": True,
            }

            if use_multi_gpu:
                kv_cache_reserve_gb = kwargs.get('kv_cache_reserve_gb', 30)
                num_gpus = torch.cuda.device_count()
                max_memory = {}
                for gpu_id in range(num_gpus):
                    total_gb = torch.cuda.get_device_properties(gpu_id).total_memory / (1024 ** 3)
                    usable_gb = int(total_gb - kv_cache_reserve_gb)
                    max_memory[gpu_id] = f"{usable_gb}GiB"
                    logger.info(f"  GPU {gpu_id}: capping weights at {usable_gb}GiB (reserving {kv_cache_reserve_gb}GB for KV cache)")
                model_kwargs["device_map"] = "balanced_low_0"
                model_kwargs["max_memory"] = max_memory
            else:
                model_kwargs["device_map"] = {"": 0}
                
            if load_in_4bit:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            elif load_in_8bit:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            else:
                model_kwargs["torch_dtype"] = torch.bfloat16
                
            use_flash_attn = kwargs.get('use_flash_attn', True)
            if use_flash_attn:
                try:
                    import flash_attn
                    model_kwargs["attn_implementation"] = "flash_attention_2"
                    if not (load_in_4bit or load_in_8bit):
                        model_kwargs["torch_dtype"] = torch.bfloat16
                    logger.info("Using Flash Attention 2")
                except ImportError:
                    logger.warning("flash-attn not installed; falling back to SDPA.")

            model_class = AutoModelForCausalLM
            is_gemma4_unified = False
            try:
                config = AutoConfig.from_pretrained(model_name, token=hf_token, trust_remote_code=True)
                if getattr(config, "model_type", None) == "gemma4_unified":
                    is_gemma4_unified = True
            except Exception:
                if os.path.isdir(model_name):
                    config_path = os.path.join(model_name, "config.json")
                    if os.path.exists(config_path):
                        try:
                            with open(config_path, "r", encoding="utf-8") as f:
                                config_dict = json.load(f)
                                if config_dict.get("model_type") == "gemma4_unified":
                                    is_gemma4_unified = True
                        except Exception:
                            pass

            if is_gemma4_unified:
                try:
                    from transformers import AutoModelForImageTextToText
                    model_class = AutoModelForImageTextToText
                except ImportError:
                    pass

            self.model = model_class.from_pretrained(model_name, token=hf_token, **model_kwargs)
        except ImportError:
            logger.error("transformers or torch not installed.")
            raise
            
    def generate(self, prompt: str, **kwargs) -> str:
        torch = self._torch
        temperature = kwargs.get('temperature', self.temperature)
        max_tokens = kwargs.get('max_tokens', self.max_tokens)
        repetition_penalty = kwargs.get('repetition_penalty', 1.0)

        messages = [{"role": "user", "content": prompt}]
        if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template is not None:
            prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt_text = prompt

        inputs = {k: v.to(self.model.device) for k, v in self.tokenizer(prompt_text, return_tensors="pt").items()}

        outputs = None
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0.0,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=repetition_penalty
                )

            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_length:]
            result = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            return result
        except Exception as e:
            logger.error(f"Local Transformers error: {e}")
            raise
        finally:
            del inputs
            if outputs is not None:
                del outputs
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            self._gc.collect()


class MockLLMInterface(LLMInterface):
    """Mock LLM interface for testing without API calls."""
    
    def __init__(self, **kwargs):
        super().__init__("mock-model", **kwargs)
        logger.info("Initialized Mock LLM interface")
    
    def generate(self, prompt: str, **kwargs) -> str:
        return "[]"


class GeminiBatchInterface(LLMInterface):
    """Interface for Google Gemini Batch API."""

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash-lite",
        api_key: Optional[str] = None,
        batch_output_dir: str = "./batch_requests",
        **kwargs,
    ):
        super().__init__(model_name, **kwargs)

        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GeminiBatchInterface requires a Gemini API key.")

        self.batch_output_dir = batch_output_dir
        self._batch_requests: List[Dict[str, Any]] = []
        self._request_id_map: Dict[str, str] = {}

        self.rpm_limit = kwargs.get('rpm_limit', 1000)
        self.min_delay = 60.0 / self.rpm_limit
        self.last_request_time = 0.0

        try:
            from google import genai
            self._rt_client = genai.Client(api_key=self.api_key)
        except ImportError:
            logger.error("google-genai not installed. Install with: pip install google-genai")
            raise

    def generate(self, prompt: str, **kwargs) -> str:
        """Submit a single prompt synchronously via the real-time API (fallback)."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_delay:
            sleep_time = self.min_delay - time_since_last
            time.sleep(sleep_time)

        temperature = kwargs.get("temperature", self.temperature)
        max_tokens  = kwargs.get("max_tokens",  self.max_tokens)
        json_schema = kwargs.get("json_schema",  None)

        gen_config: Dict[str, Any] = {
            "temperature":       temperature,
            "max_output_tokens": max_tokens,
        }
        if json_schema is not None:
            gen_config["response_mime_type"]   = "application/json"
            gen_config["response_json_schema"] = json_schema

        try:
            response = self._rt_client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=gen_config,
            )
            self.last_request_time = time.time()
            result_text = ""
            if getattr(response, "candidates", None):
                candidate = response.candidates[0]
                if getattr(candidate, "content", None) and getattr(candidate.content, "parts", None):
                    result_text = "".join(
                        p.text for p in candidate.content.parts if getattr(p, "text", None)
                    )
            if not result_text and getattr(response, "text", None):
                result_text = response.text
            if not result_text:
                raise ValueError("Model returned an empty response.")
            return result_text
        except Exception as exc:
            logger.error("GeminiBatchInterface real-time error: %s", exc)
            raise

    def queue_request(
        self,
        prompt: str,
        custom_id: str,
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> None:
        generation_config: Dict[str, Any] = {
            "temperature":       self.temperature,
            "max_output_tokens": self.max_tokens,
        }
        if json_schema:
            generation_config["response_mime_type"]   = "application/json"
            generation_config["response_json_schema"] = json_schema

        req_model_name = self.model_name
        if not (req_model_name.startswith("models/") or req_model_name.startswith("tunedModels/")):
            req_model_name = f"models/{req_model_name}"

        request = {
            "custom_id": custom_id,
            "request": {
                "model":    req_model_name,
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generation_config": generation_config,
            },
        }
        self._batch_requests.append(request)

    def flush_batch(
        self,
        display_name: str = "wiki_segmentation_batch",
        gcs_output_uri: str = "",
    ) -> str:
        if not self._batch_requests:
            raise ValueError("No requests queued. Call queue_request() first.")

        os.makedirs(self.batch_output_dir, exist_ok=True)
        jsonl_path = os.path.join(
            self.batch_output_dir,
            f"{display_name}_{int(time.time())}.jsonl",
        )
        with open(jsonl_path, "w", encoding="utf-8") as fh:
            for req in self._batch_requests:
                fh.write(json.dumps(req, ensure_ascii=False) + "\n")

        try:
            from google import genai
            from google.genai import types as _genai_types

            client = genai.Client(api_key=self.api_key)
            uploaded_file = client.files.upload(
                file=jsonl_path,
                config=_genai_types.UploadFileConfig(
                    display_name=f"llm_subj_batch_{int(time.time())}",
                    mime_type="application/jsonl"
                )
            )

            mname = self.model_name
            if not (mname.startswith("models/") or mname.startswith("tunedModels/")):
                mname = f"models/{mname}"
            batch_job_kwargs: Dict[str, Any] = {
                "model":  mname,
                "src":    uploaded_file.name,
            }
            if gcs_output_uri:
                batch_job_kwargs["dest"] = gcs_output_uri

            job = client.batches.create(**batch_job_kwargs)
            job_name: str = getattr(job, "name", str(job))
            self._batch_requests = []
            return job_name

        except Exception as exc:
            logger.error("Failed to submit Gemini Batch job: %s", exc)
            raise

    def check_batch_status(self, job_name: str) -> Dict[str, Any]:
        try:
            from google import genai
            client = genai.Client(api_key=self.api_key)
            job = client.batches.get(name=job_name)
            state = getattr(job, "state", "UNKNOWN")
            if hasattr(state, "name"):
                state_str = state.name
            else:
                state_str = str(state)
            if "." in state_str:
                state_str = state_str.split(".")[-1]
            return {"name": job_name, "state": state_str, "job": job}
        except Exception as exc:
            logger.error("Failed to check batch status: %s", exc)
            raise

    def download_batch_results(
        self,
        job_name: str,
        output_dir: str,
    ) -> List[Dict[str, Any]]:
        try:
            from google import genai
            client = genai.Client(api_key=self.api_key)
            job = client.batches.get(name=job_name)

            results_uri = getattr(job, "dest", None) or getattr(job, "output_uri", None)
            if not results_uri:
                raise ValueError("Batch job results URI not found. Check if the job has completed.")

            os.makedirs(output_dir, exist_ok=True)
            local_path = os.path.join(output_dir, f"{job_name.replace('/', '_')}_results.jsonl")

            file_name = getattr(results_uri, "file_name", None)
            if file_name:
                content = client.files.download(file=file_name)
                with open(local_path, "wb") as fh:
                    fh.write(content)
            else:
                gcs_uri = getattr(results_uri, "gcs_uri", None)
                gcs_uri_str = None
                if isinstance(gcs_uri, list) and gcs_uri:
                    gcs_uri_str = gcs_uri[0]
                elif isinstance(gcs_uri, str):
                    gcs_uri_str = gcs_uri
                elif isinstance(results_uri, str) and results_uri.startswith("gs://"):
                    gcs_uri_str = results_uri

                if gcs_uri_str:
                    try:
                        from google.cloud import storage as gcs
                        bucket_name, blob_prefix = gcs_uri_str.replace("gs://", "").split("/", 1)
                        gcs_client = gcs.Client()
                        with open(local_path, "wb") as fh:
                            for blob in gcs_client.list_blobs(bucket_name, prefix=blob_prefix):
                                blob.download_to_file(fh)
                    except ImportError:
                        raise ImportError("google-cloud-storage is required to download batch results.")
                else:
                    raise RuntimeError(f"Unsupported results destination: {results_uri}")

            results = []
            with open(local_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            results.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
            return results

        except Exception as exc:
            logger.error("Failed to download batch results: %s", exc)
            raise


def create_llm_interface(
    backend: str,
    model_name: str,
    api_key: Optional[str] = None,
    **kwargs
) -> LLMInterface:
    """
    Factory function to create appropriate LLM interface.

    Args:
        backend: Backend type.
                 Supported: ``huggingface``, ``gemini``, ``openai``,
                 ``gemini_batch``, ``mock``, ``local_transformers``.
        model_name: Model identifier.
        api_key: API key if required.
        **kwargs: Additional backend-specific parameters.

    Returns:
        LLMInterface instance.
    """
    backend = backend.lower()

    if backend == "huggingface":
        return HuggingFaceInterface(model_name, api_key, **kwargs)
    elif backend == "openai":
        return OpenAIInterface(model_name, api_key, **kwargs)
    elif backend == "gemini":
        return GeminiInterface(model_name, api_key, **kwargs)
    elif backend == "gemini_batch":
        return GeminiBatchInterface(model_name, api_key, **kwargs)
    elif backend == "mock":
        return MockLLMInterface(**kwargs)
    elif backend == "local_transformers":
        return LocalTransformersInterface(model_name, **kwargs)
    else:
        raise ValueError(
            f"Unsupported backend: {backend!r}. "
            "Choose from: huggingface, gemini, gemini_batch, openai, mock, local_transformers"
        )
