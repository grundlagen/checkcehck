import os
import sys
import subprocess
import time
import threading
import requests
import json
import traceback
import re
import difflib
import shutil
from collections import deque

# Configuration
MAX_ATTEMPTS = 15
BASE_FILE_PATH = r"C:\Users\Lenovo\.spyder-py3\xtts_match_embed.py"
DEEPSEEK_API_KEY = ""
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEBUG_LOG = "autofix_debug.log"
CUMULATIVE_KNOWLEDGE_FILE = "autofix_cumulative_knowledge.txt"
# Then run normally:
python "C:\Users\Lenovo\.spyder-py3\xtts_match_embed.py"
# Logging setup
def log_debug(message):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] {message}"
    print(entry)
    with open(DEBUG_LOG, "a", encoding="utf-8") as log:
        log.write(entry + "\n")

# History tracking
PATCH_HISTORY = deque(maxlen=MAX_ATTEMPTS)
ERROR_HISTORY = deque(maxlen=MAX_ATTEMPTS)
ORIGINAL_CODE = None
CURRENT_VERSION = 0

def get_versioned_file_path(version):
    """Get versioned file path"""
    base, ext = os.path.splitext(BASE_FILE_PATH)
    return f"{base}_v{version}{ext}"

def capture_xtts_debug_info():
    """Capture detailed debug information from XTTS components"""
    debug_info = []
    try:
        # Try to get version info
        import torch
        import TTS
        debug_info.append(f"PyTorch version: {torch.__version__}")
        debug_info.append(f"TTS version: {TTS.__version__}")
        
        # Try to get GPU info
        if torch.cuda.is_available():
            debug_info.append(f"GPU: {torch.cuda.get_device_name(0)}")
            debug_info.append(f"CUDA version: {torch.version.cuda}")
        else:
            debug_info.append("CUDA not available")
            
        # Try to get model config info
        try:
            from TTS.tts.configs.xtts_config import XttsConfig
            config = XttsConfig()
            debug_info.append(f"XTTSConfig: {config}")
        except Exception as e:
            debug_info.append(f"Config error: {str(e)}")
            
    except ImportError:
        debug_info.append("TTS/torch not importable")
    
    return "\n".join(debug_info)

def run_script(version):
    """Execute the target script and monitor its output"""
    file_path = get_versioned_file_path(version)
    log_debug(f"Starting script: {file_path}")
    process = subprocess.Popen(
        ["python", file_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    output_lines = []
    start_time = time.time()
    last_update = time.time()
    update_interval = 600  # 10 minutes

    def watchdog():
        """Monitor script execution for hangs"""
        nonlocal last_update
        while process.poll() is None:
            if time.time() - last_update > update_interval:
                log_msg = f"[WATCHDOG] Script running: {int(time.time() - start_time)}s"
                log_debug(log_msg)
                last_update = time.time()
            time.sleep(10)

    watchdog_thread = threading.Thread(target=watchdog, daemon=True)
    watchdog_thread.start()

    # Stream output in real-time
    for line in process.stdout:
        line = line.strip()
        if line:
            output_lines.append(line)
            last_update = time.time()
            print(f"[xtts] {line}")

    process.wait()
    combined_output = "\n".join(output_lines)
    log_debug(f"Script exited with code: {process.returncode}")

    # Enhanced failure patterns
    failure_signatures = [
        "XTTS load error",
        "TTS disabled or failed to load",
        "No module named",
        "ImportError",
        "ModuleNotFoundError",
        "XTTS model status: not loaded",
        "WeightsUnpickler error",
        "weights_only",
        "AttributeError",
        "TypeError",
        "ValueError",
        "CUDA error",
        "out of memory",
        "torch.cuda",
        "falling back to CPU",
        "RuntimeError",
        "unterminated string literal",
        "SyntaxError",
        "IndentationError"
    ]

    if any(sig in combined_output for sig in failure_signatures):
        log_debug("Failure signature detected")
        return False, combined_output

    return process.returncode == 0, combined_output

def analyze_error_patterns():
    """Analyze error history for patterns"""
    if not ERROR_HISTORY:
        return "", False
    
    analysis = []
    radical = False
    
    # GPU/CPU oscillation detection
    gpu_errors = sum(1 for e in ERROR_HISTORY if "CUDA" in e or "out of memory" in e)
    cpu_errors = sum(1 for e in ERROR_HISTORY if "falling back to CPU" in e)
    if gpu_errors >= 2 and cpu_errors >= 2:
        analysis.append("Oscillating between GPU and CPU errors")
        radical = True
    
    # Weights loading issues
    weights_errors = sum(1 for e in ERROR_HISTORY if "WeightsUnpickler" in e or "weights_only" in e)
    if weights_errors >= 2:
        analysis.append("Persistent model weights loading issues")
        radical = True
    
    # Syntax errors in fixes
    syntax_errors = sum(1 for e in ERROR_HISTORY if any(
        err in e for err in ["unterminated string", "SyntaxError", "IndentationError"]
    ))
    if syntax_errors >= 1:
        analysis.append("Previous fixes contained syntax errors")
        radical = True
    
    return "\n".join(analysis), radical

def clean_api_response(response_text):
    """Robust cleaning of API response to prevent syntax errors"""
    # Remove markdown code blocks
    if response_text.startswith("```python"):
        response_text = response_text[9:].rsplit("```", 1)[0]
    elif response_text.startswith("```"):
        response_text = response_text[3:].rsplit("```", 1)[0]
    
    # Remove JSON artifacts if present
    if response_text.startswith("{") and response_text.endswith("}"):
        try:
            json_data = json.loads(response_text)
            if "content" in json_data:
                response_text = json_data["content"]
        except:
            pass
    
    # Remove non-code headers/footers
    lines = response_text.split("\n")
    code_lines = []
    in_code = False
    
    for line in lines:
        if re.match(r"^[a-zA-Z0-9_]+\.py", line):
            continue  # Skip file name headers
        if "Repaired code:" in line:
            continue
        if line.strip().startswith("#") or line.strip().startswith('"'):
            code_lines.append(line)
        else:
            if any(keyword in line for keyword in ["import ", "from ", "def ", "class ", " = "]):
                in_code = True
            if in_code:
                code_lines.append(line)
    
    cleaned_code = "\n".join(code_lines)
    
    # Ensure it starts with valid Python syntax
    if not cleaned_code.startswith(("import ", "from ", "#", '"', "'", "def ", "class ")):
        # Find first valid Python line
        for i, line in enumerate(lines):
            if line.strip() and not line.strip().startswith(("```", "Repaired", "###")):
                cleaned_code = "\n".join(lines[i:])
                break
    
    return cleaned_code.strip()

def handle_user_interaction(fixed_code):
    """Pause for user input if the API requests it"""
    if "[USER_INPUT_NEEDED]" in fixed_code:
        print("\n=== USER INPUT REQUIRED ===")
        question_block = fixed_code.split("[USER_INPUT_NEEDED]")[1].split("[END_OPTIONS]")[0]
        print(question_block)
        response = input("Your choice (enter letter): ").strip().upper()
        
        # Add the response to the context
        user_context = f"\n# USER RESPONSE: {response}\n"
        return fixed_code.replace("[USER_INPUT_NEEDED]", user_context)
    return fixed_code

def update_cumulative_knowledge(attempt, error, patch):
    """Maintain a rolling summary of key learnings"""
    # Initialize if needed
    global ORIGINAL_CODE
    if ORIGINAL_CODE is None:
        with open(BASE_FILE_PATH, "r", encoding="utf-8") as f:
            ORIGINAL_CODE = f.read()
    
    # Create patch summary
    patch_summary = ""
    if patch:
        # Compare to original to find key changes
        original_lines = ORIGINAL_CODE.splitlines()
        current_lines = patch.splitlines()
        diff = difflib.unified_diff(original_lines, current_lines, n=3)
        
        # Extract key changes
        key_changes = []
        for line in diff:
            if line.startswith('+') and not line.startswith('+++'):
                key_changes.append(line)
            if len(key_changes) >= 5:  # Limit to 5 most relevant changes
                break
        patch_summary = "\n".join(key_changes[:5])
    
    # Update every 5 attempts or when empty
    if attempt % 5 == 0 or not os.path.exists(CUMULATIVE_KNOWLEDGE_FILE):
        with open(CUMULATIVE_KNOWLEDGE_FILE, "a", encoding="utf-8") as f:
            f.write(f"=== ATTEMPT {attempt} SUMMARY ===\n")
            f.write(f"Key error: {error[:300]}\n")
            f.write(f"Key changes:\n{patch_summary}\n\n")
    
    # Read cumulative knowledge
    try:
        with open(CUMULATIVE_KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""

def autofix_code(code, error_msg, attempt_number):
    """Use DeepSeek API to repair Python code with context"""
    log_debug(f"Autofix initiated | Attempt: {attempt_number}")
    
    # Analyze error patterns
    pattern_analysis, radical_mode = analyze_error_patterns()
    
    # Capture debug info
    xtts_debug = capture_xtts_debug_info()
    
    # Update cumulative knowledge
    cumulative_knowledge = update_cumulative_knowledge(
        attempt_number, error_msg, code
    )
    
    # Enhanced system prompt
    system_prompt = f"""
You are CodeMedic v5.0, an advanced Python repair agent. Fixing `xtts_match_embed.py` for Coqui XTTS v2.

# CUMULATIVE KNOWLEDGE (key learnings):
{cumulative_knowledge[:1500]}{'...' if len(cumulative_knowledge) > 1500 else ''}

# CURRENT SITUATION:
Attempt: {attempt_number}/{MAX_ATTEMPTS}
Error: {error_msg[:300]}{'...' if len(error_msg) > 300 else ''}
Debug: {xtts_debug}
Patterns: {pattern_analysis}

# IMMEDIATE ACTION PLAN:
1. FIX WEIGHTS LOADING:
   - Use: torch.serialization.add_safe_globals([XttsConfig])
2. HANDLE GPU/CPU:
   - Robust device detection
   - Explicit tensor.to(device)
3. PRESERVE ORIGINAL FUNCTIONALITY
4. AVOID SYNTAX ERRORS

# RETURN ONLY COMPLETE, VALID PYTHON CODE
"""
    if radical_mode:
        system_prompt += "\n# RADICAL MODE: Restructure problematic sections\n"

    # Construct API payload
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Repair this code:\n{code}"}
    ]
    
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
        "User-Agent": "CodeMedic-Autofix/5.0"
    }
    
    payload = {
        "model": "deepseek-reasoner",
        "messages": messages,
        "temperature": 0.5,
        "max_tokens": 4096,
        "stop": ["<|endoftext|>", "###", "```"]
    }
    
    try:
        log_debug(f"Sending API request | Radical: {radical_mode}")
        start_api = time.time()
        
        response = requests.post(
            DEEPSEEK_API_URL,
            headers=headers,
            json=payload,
            timeout=120
        )
        
        api_time = time.time() - start_api
        log_debug(f"API response: {response.status_code} | Time: {api_time:.2f}s")
        
        if response.status_code != 200:
            log_debug(f"API error: {response.text[:300]}")
            return code, radical_mode
        
        response_data = response.json()
        raw_response = response_data['choices'][0]['message']['content']
        
        # Clean response
        fixed_code = clean_api_response(raw_response)
        log_debug(f"Cleaned code | Length: {len(fixed_code)} chars")
        
        # Handle user interaction if needed
        fixed_code = handle_user_interaction(fixed_code)
        
        return fixed_code, radical_mode
        
    except Exception as e:
        log_debug(f"API failed: {str(e)}")
        traceback.print_exc()
        return code, False

def validate_python_syntax(code):
    """Validate Python syntax before applying changes"""
    try:
        compile(code, "temp.py", "exec")
        return True, ""
    except SyntaxError as e:
        error_msg = f"Syntax error: {e.msg} at line {e.lineno}"
        return False, error_msg
    except Exception as e:
        return False, str(e)

def main_loop():
    """Main repair loop with versioned files"""
    global ORIGINAL_CODE, CURRENT_VERSION
    
    log_debug("===== AUTOFIX SESSION STARTED =====")
    attempt = 0
    
    # Create initial versioned file
    versioned_path = get_versioned_file_path(CURRENT_VERSION)
    shutil.copyfile(BASE_FILE_PATH, versioned_path)
    log_debug(f"Created initial version: {versioned_path}")
    
    # Load original code
    with open(versioned_path, "r", encoding="utf-8") as f:
        ORIGINAL_CODE = f.read()
    
    while attempt < MAX_ATTEMPTS:
        attempt += 1
        log_debug(f"\n=== ATTEMPT {attempt}/{MAX_ATTEMPTS} ===")
        
        # Execute and monitor the target script
        success, output = run_script(CURRENT_VERSION)
        if success:
            log_debug("SUCCESS: Script completed without errors")
            print("Script ran successfully.")
            return
            
        # Store error for pattern analysis
        ERROR_HISTORY.append(output)
        
        # Save full output to file
        with open(f"attempt_{attempt}_output.log", "w", encoding="utf-8") as log_file:
            log_file.write(output)
        
        # Load current code version
        current_file = get_versioned_file_path(CURRENT_VERSION)
        with open(current_file, "r", encoding="utf-8") as f:
            current_code = f.read()
        
        try:
            # Generate repaired code with full context
            fixed_code, radical_used = autofix_code(current_code, output, attempt)
            
            # Store the patch attempt
            PATCH_HISTORY.append(fixed_code[:2000])
            
            # Validate syntax before applying
            is_valid, error_msg = validate_python_syntax(fixed_code)
            
            if not is_valid:
                log_debug(f"Syntax validation FAILED: {error_msg}")
                with open(f"invalid_attempt_{attempt}.py", "w", encoding="utf-8") as f_invalid:
                    f_invalid.write(fixed_code)
                log_debug("Skipping invalid fix")
                continue
            
            # Create new version
            CURRENT_VERSION += 1
            new_version_file = get_versioned_file_path(CURRENT_VERSION)
            with open(new_version_file, "w", encoding="utf-8") as f:
                f.write(fixed_code)
            log_debug(f"Created version {CURRENT_VERSION}: {new_version_file}")
                
        except Exception as e:
            log_debug(f"Repair process failed: {str(e)}")
            traceback.print_exc()
            
    # Final failure handling
    log_debug("FAILED: Maximum attempts reached")
    with open("autofix_fail_report.txt", "w", encoding="utf-8") as f:
        f.write(f"After {MAX_ATTEMPTS} attempts:\n\n")
        f.write("=== ERROR HISTORY ===\n")
        for i, error in enumerate(ERROR_HISTORY, 1):
            f.write(f"\nAttempt {i} error:\n{error[:1000]}...\n")
        
        f.write("\n=== PATCH HISTORY ===\n")
        for i, patch in enumerate(PATCH_HISTORY, 1):
            f.write(f"\nAttempt {i} patch:\n{patch[:1000]}...\n")
    
    print("Too many failures. See autofix_fail_report.txt for full history.")

if __name__ == "__main__":
    main_loop()