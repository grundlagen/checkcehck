import os
import subprocess
import time
import threading
import requests
import json
import traceback

# Number of max attempts before giving up
MAX_ATTEMPTS = 15

# Path to the target script to fix
FILE_PATH = r"C:\Users\Lenovo\.spyder-py3\xtts_match_embed.py"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    raise RuntimeError("DEEPSEEK_API_KEY environment variable is not set")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

# Enhanced logging setup
DEBUG_LOG = "autofix_debug.log"
def log_debug(message):
    """Log debug information with timestamp"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] {message}"
    print(entry)
    with open(DEBUG_LOG, "a", encoding="utf-8") as log:
        log.write(entry + "\n")

# Store patch history for context
PATCH_HISTORY = []

def run_script():
    """Execute the target script and monitor its output"""
    log_debug(f"Starting script: {FILE_PATH}")
    process = subprocess.Popen(
        ["python", FILE_PATH],
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

    # Known failure patterns
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
        "ValueError"
    ]

    if any(sig in combined_output for sig in failure_signatures):
        log_debug("Failure signature detected")
        return False, combined_output

    return process.returncode == 0, combined_output

def autofix_code(code, error_msg, radical_mode=False):
    """Use DeepSeek API to repair Python code"""
    log_debug(f"Autofix initiated | Radical mode: {radical_mode}")
    
    # Provide context of previous patches
    patch_context = "\n# PREVIOUS PATCH ATTEMPTS:\n"
    if PATCH_HISTORY:
        for i, patch in enumerate(PATCH_HISTORY[-3:]):  # Show last 3 patches
            patch_context += f"# ATTEMPT {len(PATCH_HISTORY)-3+i}:\n{patch[:500]}...\n\n"
    else:
        patch_context += "# No previous patches\n"

    # Enhanced system prompt with versioning
    system_prompt = f"""
You are CodeMedic v2.5, a fault-tolerant Python repair agent. You're repairing `xtts_match_embed.py` which extracts phonetic embeddings using Coqui XTTS v2.

# CONTEXT:
Current date: {time.strftime("%Y-%m-%d")}
Target environment: Windows 11, Python 3.9, PyTorch 2.0+
Previous error: {error_msg[:500]}{'...' if len(error_msg) > 500 else ''}

{patch_context}

# OBJECTIVE:
Make the script run end-to-end without crashing. Ensure embeddings are extracted successfully.

# RULES:
1. RETURN ONLY VALID PYTHON CODE - no markdown, no explanations
2. Immediately fix syntax errors, import errors, and attribute errors
3. Remove any non-code text at the top of the file
4. For model loading issues, use:
    from TTS.tts.configs.xtts_config import XttsConfig
    import torch
    torch.serialization.add_safe_globals([XttsConfig])  # NOTE: 'add_safe_globals' not 'add_safe_class'
5. Preserve GPU fallback to CPU logic
6. Add missing imports with install commands if needed
7. Add progress logging if script might hang
8. Handle Windows path issues (backslashes, drive letters)
9. If radical mode: rewrite faulty sections completely
"""
    if radical_mode:
        system_prompt += "\n# RADICAL MODE: Restructure problematic sections completely\n"

    # Construct API payload
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Repair this code:\n{code}"}
    ]
    
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
        "User-Agent": "CodeMedic-Autofix/1.0"
    }
    
    payload = {
        "model": "deepseek-chat",  # CORRECTED MODEL NAME
        "messages": messages,
        "temperature": 0.8,  # Optimal balance (0-2 range)
        "max_tokens": 4096
    }
    
    try:
        # API call with detailed diagnostics
        log_debug(f"Sending API request | Tokens: ~{len(json.dumps(payload))//4}")
        log_debug(f"Prompt context: {system_prompt[:300]}...")  # Show part of prompt
        start_api = time.time()
        
        response = requests.post(
            DEEPSEEK_API_URL,
            headers=headers,
            json=payload,
            timeout=120  # Longer timeout for complex fixes
        )
        
        api_time = time.time() - start_api
        log_debug(f"API response: {response.status_code} | Time: {api_time:.2f}s")
        
        if response.status_code != 200:
            log_debug(f"API error: {response.text[:300]}")
            return code  # Fallback to original
        
        response_data = response.json()
        fixed_code = response_data['choices'][0]['message']['content']
        
        # Clean and validate response
        fixed_code = fixed_code.strip("`").replace("python", "").strip()
        log_debug(f"Received fixed code | Length: {len(fixed_code)} chars")
        
        # Store patch for historical context
        PATCH_HISTORY.append(fixed_code[:1000])  # Store first 1000 chars
        
        return fixed_code
        
    except Exception as e:
        log_debug(f"API failed: {str(e)}")
        traceback.print_exc()
        return code

def main_loop():
    """Main repair loop with enhanced diagnostics"""
    log_debug("===== AUTOFIX SESSION STARTED =====")
    last_errors = []
    attempt = 0
    
    while attempt < MAX_ATTEMPTS:
        attempt += 1
        log_debug(f"\n=== ATTEMPT {attempt}/{MAX_ATTEMPTS} ===")
        
        # Execute and monitor the target script
        success, output = run_script()
        if success:
            log_debug("SUCCESS: Script completed without errors")
            print("Script ran successfully.")
            return
            
        # Error analysis
        error_freq = last_errors.count(output.strip())
        radical_mode = error_freq >= 2
        last_errors.append(output.strip())
        if len(last_errors) > 5:
            last_errors.pop(0)
            
        log_debug(f"ERROR detected | Radical: {radical_mode}\n{output[:500]}...")
        
        # Load current code version
        with open(FILE_PATH, "r", encoding="utf-8") as f:
            current_code = f.read()
        
        try:
            # Generate repaired code
            fixed_code = autofix_code(current_code, output, radical_mode)
            
            # Create backup
            backup_path = f"{FILE_PATH}.bak{attempt}"
            with open(backup_path, "w", encoding="utf-8") as b:
                b.write(current_code)
            log_debug(f"Backup created: {backup_path}")
            
            # Validate syntax before applying
            try:
                compile(fixed_code, FILE_PATH, "exec")
                with open(FILE_PATH, "w", encoding="utf-8") as f:
                    f.write(fixed_code)
                log_debug("Applied repaired code")
                
            except SyntaxError as e:
                log_debug(f"Syntax check FAILED: {str(e)}")
                with open("invalid_fix.py", "w", encoding="utf-8") as err_file:
                    err_file.write(fixed_code)
                
        except Exception as e:
            log_debug(f"Repair process failed: {str(e)}")
            traceback.print_exc()
            
    # Final failure handling
    log_debug("FAILED: Maximum attempts reached")
    with open("autofix_fail_report.txt", "w", encoding="utf-8") as f:
        f.write(f"After {MAX_ATTEMPTS} attempts:\n\nLast error:\n{output}")
    print("Too many failures. Manual intervention needed.")

if __name__ == "__main__":
    main_loop()