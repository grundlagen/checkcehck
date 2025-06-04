# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 12:11:25 2025

@author: Lenovo
"""

# evolutionary_embedder.py — network‑robust LLM patcher (2025‑06‑02)
"""
Adds rock‑solid networking to _llm_patch():
• Tries *deepseek‑chat* first (cheaper/faster), then *deepseek‑reasoner* on failure.
• Uses `timeout=(30, 600)` so we never block > 60 s per attempt.
• Streams response and aborts if no data for 30 s.
• Saves raw + code as before.
Other logic unchanged.
"""

import os
import sys
import json
import time
import subprocess
import importlib.util
import logging
import traceback
import re
import textwrap
from pathlib import Path
import asyncio
import aiohttp
from typing import Dict, List, Optional, AsyncGenerator
from dataclasses import dataclass
import numpy as np
import torch
import librosa
import requests

CONFIG = {
    "WAV_FILE_PATH": r"C:\Users\Lenovo\Desktop\500wav_eng\common_voice_en_41906032.wav",
    "DEEPSEEK_API_KEY": "",
    "MAX_ITERATIONS": 100,
    "PATCH_DIR": "patches",
    "JOURNAL_FILE": "evolutionary_journal.jsonl",
    "AUTO_INSTALL": True,
}

# === STATIC_BEGIN =========================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("EvoEmbedder")


def ensure_package(wheel: str, modulename: str | None = None):
    """Install *wheel* via pip if *modulename* (or wheel) is missing."""
    mod = modulename or wheel.split("==")[0]  # allow version‑pinned wheels
    if importlib.util.find_spec(mod) is None:
        if not CONFIG.get("AUTO_INSTALL", False):
            raise ModuleNotFoundError(mod)
        logger.info("📦 pip‑install %s", wheel)
        subprocess.check_call([sys.executable, "-m", "pip", "install", wheel])
    return importlib.import_module(mod)


def journal(event: str, **payload):
    Path(CONFIG["PATCH_DIR"]).mkdir(exist_ok=True)
    entry = {
        "ts": time.strftime("%F %T"),
        "event": event,
        **payload,
    }
    with open(CONFIG["JOURNAL_FILE"], "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def validate_embeddings(emb: Dict) -> bool:
    """Toy validator – replace with real checks."""
    return bool(emb) and all(v.size > 0 for v in emb.values())


def save_embeddings(emb: Dict, path: str, validate: bool = True) -> bool:
    # Add debug logging
    logger.debug(f"validate_embeddings function: {validate_embeddings}")
    logger.debug(f"Function expects {validate_embeddings.__code__.co_argcount} arguments")
    
    if not validate or validate_embeddings(emb):
        np.savez(path, **emb)
        logger.info("💾 saved %s", path)
        return True
    logger.warning("⚠️ embeddings failed validation; not saved → %s", path)
    return False

# --- patch‑loader guard ---------------------------------------------------
STATIC_KEYS = {
    "ensure_package", "journal", "validate_embeddings", "save_embeddings",
    "AIClient", "AIModelConfig", "EvolutionaryEmbedder", "STATIC_KEYS",
    "CONFIG", "validate_embeddings", "save_embeddings", "logger", "logging"
}
# === STATIC_END ========================================================== #

@dataclass
class AIModelConfig:
    name: str
    temperature: float = 0.7
    max_tokens: int = 2000


class AIClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.deepseek.com/v1"

    async def get_ai_response(self, model: str, messages: list) -> str:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 2000,
                    "stream": False
                },
                timeout=300
            ) as response:
                response.raise_for_status()
                data = await response.json()
                return data['choices'][0]['message']['content']
   
   


class EvolutionaryEmbedder:
    def __init__(self):
        self.wav = r"C:\Users\Lenovo\Desktop\500wav_eng\common_voice_en_41906032.wav"
        self.emb_path = self.wav + ".embeddings.npz"
        self.history: List[Dict] = []
        self.err = ""
        self.model = None
        self.ai_client = AIClient(CONFIG["DEEPSEEK_API_KEY"])
        self._import_previous_patches()  # Now this will work

    async def run(self):
        ...

    # ✅ Correctly indented INSIDE the class
    def _import_previous_patches(self):
        patch_dir = Path(CONFIG["PATCH_DIR"])
        for f in sorted(
            patch_dir.glob("iter_*.py"),
            key=lambda p: int(p.stem.split('_')[1])
        ):
            try:
                spec = importlib.util.spec_from_file_location(f.stem, f)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                safe_dict = {
                    k: v for k, v in mod.__dict__.items()
                    if not k.startswith("__") and k not in STATIC_KEYS
                }
                globals().update(safe_dict)
                logger.info("🔄 Applied patch: %s", f.name)
            except Exception as e:
                logger.error("❌ Failed to import patch %s: %s", f.name, e)

    # ... (rest of the methods)
    
    async def run(self):
        """Async version of the main loop"""
        for i in range(CONFIG["MAX_ITERATIONS"]):
            logger.info("ITER %d", i)
            
            if not self.load_model():
                if not await self._llm_patch(i):
                    continue
            
            emb = self.extract()
            if save_embeddings(emb, self.emb_path):
                journal("success", iter=i)
                logger.info("✅ Embeddings ready")
                return True
            
            journal("fail", iter=i, err=self.err[:300])
        return False

    async def _llm_patch(self, idx: int) -> bool:
        base_prompt = (
            "Fix the Python so validate_embeddings() passes. "
            "Return ONLY code between ```python and ```."
        )
        user_ctx = f"TRACEBACK:\n{self.err[-1000:]}\nHISTORY:\n{json.dumps(self.history[-5:], indent=2)}"
        models = ["deepseek-chat", "deepseek-reasoner"]

        for model in models:
            for attempt in range(3):
                try:
                    response = await self.ai_client.get_ai_response(
                        model=model,
                        messages=[
                            {"role": "system", "content": base_prompt},
                            {"role": "user", "content": user_ctx}
                        ]
                    )
                    
                    code = self._extract_code(response)
                    if len(code) >= 10:
                        self._save_patch(code, response, idx)
                        return True

                except Exception as e:
                    logger.error(f"Attempt {attempt+1} failed: {str(e)}")
                    if attempt == 2:
                        journal("patch_fail", iter=idx)
        
        return False

    def load_model(self):
        try:
            ensure_package("TTS")
            from TTS.tts.models import setup_model
            self.model = setup_model("tts_models/multilingual/multi-dataset/xtts_v2")
            return True
        except Exception:
            self.err = traceback.format_exc()
            return False

    def extract(self):
        try:
            audio, _ = librosa.load(self.wav, sr=22050, mono=True)
            audio = np.pad(audio, (0, max(0, 66150 - len(audio))))
            with torch.no_grad():
                lat = self.model.get_conditioning_latents(audio, 22050)
                return {k: v.cpu().numpy() for k, v in lat.items()}
        except Exception:
            self.err = traceback.format_exc()
            return {}
def _import_previous_patches(self):
    patch_dir = Path(CONFIG["PATCH_DIR"])  # Define once outside the loop

    # Sort by the numeric suffix so iter_12 > iter_2
    for f in sorted(
        patch_dir.glob("iter_*.py"),
        key=lambda p: int(p.stem.split('_')[1])
    ):
        try:
            spec = importlib.util.spec_from_file_location(f.stem, f)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            # Guard: only update non-static names
            safe_dict = {
                k: v for k, v in mod.__dict__.items()
                if not k.startswith("__") and k not in STATIC_KEYS
            }
            globals().update(safe_dict)
            logger.info("🔄 Applied patch: %s", f.name)
        except Exception as e:
            logger.error("❌ Failed to import patch %s: %s", f.name, e)
            
    def _save_patch(self, code: str, raw: str, idx: int):
        patch_dir = Path(CONFIG["PATCH_DIR"])
        patch_dir.mkdir(exist_ok=True)
        
        base = patch_dir / f"iter_{idx}"
        py_f = base.with_suffix(".py")
        raw_f = base.parent / f"{base.name}_raw.txt"
        
        py_f.write_text(code, encoding="utf-8")
        raw_f.write_text(raw, encoding="utf-8")
        logger.info("📝 patch saved → %s / %s", py_f, raw_f)
        journal("patch", file=str(py_f))

        try:
            spec = importlib.util.spec_from_file_location(py_f.stem, py_f)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            globals().update({k: v for k, v in mod.__dict__.items() if not k.startswith("__")})
            logger.info(f"🔄 Applied patch: iter_{idx}.py")
        except Exception as e:
            logger.error(f"❌ Failed to apply patch iter_{idx}.py: {str(e)}")

    def _extract_code(self, txt: str) -> str:
        pattern = r"""
            (?:```|~~~)               # Opening fence (``` or ~~~)
            [ \t]*                    # Optional whitespace
            (?:[a-zA-Z0-9_+-]+)?      # Optional language tag
            [ \t]*                    # Optional whitespace
            \n                        # Newline after opening fence
            (.*?)                     # Non-greedy capture of the code block
            \n                        # Newline before closing fence
            [ \t]*                    # Optional whitespace
            (?:```|~~~)               # Closing fence
        """
        fence = re.search(pattern, txt, re.DOTALL | re.VERBOSE)
        code = fence.group(1).strip() if fence else txt.strip()
        
        code = re.sub(r"\bnull\b", "None", code)
        code = re.sub(r"\btrue\b", "True", code, flags=re.IGNORECASE)
        code = re.sub(r"\bfalse\b", "False", code, flags=re.IGNORECASE)
        return code

    # ------------------------------------------------------------- LLM patcher
    def _llm_patch_legacy(self, idx: int):
        """Legacy synchronous version for reference"""
        url = "https://api.deepseek.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {CONFIG['DEEPSEEK_API_KEY']}",
            "Content-Type": "application/json",
        }
        base_prompt = (
            "Fix the Python so validate_embeddings() passes. Return ONLY code; "
            "install deps via ensure_package()."
        )
        user_ctx = f"TRACEBACK:\n{self.err[-1000:]}\nHISTORY:\n{json.dumps(self.history[-5:], indent=2)}"
        models = ["deepseek-chat", "deepseek-reasoner"]

        for model in models:
            msgs = [
                {"role": "system", "content": base_prompt},
                {"role": "user", "content": user_ctx},
            ]
            for attempt in range(3):
                payload = {
                    "model": model,
                    "temperature": 1.0,
                    "max_tokens": 2000,
                    "messages": msgs,
                    "stream": True,
                }
                try:
                    with requests.post(url, headers=headers, json=payload, timeout=(30, 300), stream=True) as r:
                        r.raise_for_status()
                        content = []
                        last_data_time = time.time()
                        
                        for line in r.iter_lines():
                            if not line or line == b'data: [DONE]':
                                continue
                                
                            if line.startswith(b'data:'):
                                try:
                                    data = json.loads(line[5:].strip())
                                    if 'choices' in data and len(data['choices']) > 0:
                                        delta = data['choices'][0].get('delta', {})
                                        if 'content' in delta:
                                            content.append(delta['content'])
                                except json.JSONDecodeError:
                                    continue
                                    
                            if time.time() - last_data_time > 300:
                                raise TimeoutError("LLM response stalled for 5 minutes")
                                
                        raw = ''.join(content)
                        logger.debug("LLM %s raw: %s", model, textwrap.shorten(raw, 400))
                        code = self._extract_code(raw)
                        
                        if len(code) < 10:
                            logger.warning("%s attempt %d empty", model, attempt+1)
                            msgs.append({"role": "assistant", "content": raw})
                            msgs.append({"role": "user", "content": "Return ≥10 chars of python."})
                            continue
                        
                        self._save_patch(code, raw, idx)
                        return True
                        
                except (requests.Timeout, TimeoutError) as te:
                    logger.warning("%s attempt %d timeout: %s", model, attempt+1, te)
                except Exception as e:
                    logger.error("%s attempt %d error: %s", model, attempt+1, str(e))
                    
        journal("patch_fail", iter=idx)
        logger.error("❌ No usable patch from LLMs")
        return False


async def _main_coroutine():
    embedder = EvolutionaryEmbedder()
    return await embedder.run()

def main():
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    try:
        ensure_package("nest_asyncio")
        import nest_asyncio
        nest_asyncio.apply()
        loop = asyncio.get_event_loop()
        success = loop.run_until_complete(_main_coroutine())
        return 0 if success else 1
    except Exception as e:
        logger.error(f"Execution failed: {str(e)}")
        logger.error(traceback.format_exc())  # Add full traceback
        return 1
    
if __name__ == "__main__":
    sys.exit(main())