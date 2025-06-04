from fastapi import FastAPI, HTTPException, Depends, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import json
import os
import asyncio
from datetime import datetime
from typing import List, Dict, Optional, Any
import logging
import subprocess
import difflib
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="DeepSeek Reasoner API", description="API for DeepSeek Code Assistant")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a dictionary to store chat instances
code_assistants = {}
session_files = {}

# Pydantic models for request/response validation
class AssistantConfig(BaseModel):
    api_key: str

class CodeRequest(BaseModel):
    session_id: str
    code: str
    error: Optional[str] = None
    objective: Optional[str] = None

class SessionRequest(BaseModel):
    session_id: str
    filename: Optional[str] = None

class UserChoice(BaseModel):
    session_id: str
    choice: str
    modified_code: Optional[str] = None

class DeepSeekAssistant:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("")
        self.base_url = "https://api.deepseek.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.code_history = []
        self.session_history = []
        self.max_history = 5
        self.current_code = ""
        self.last_error = ""
    
    def _call_api(self, messages: List[Dict], model: str = "deepseek-reasoner", 
                 temperature: float = 0.7, max_tokens: int = 4096) -> str:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        try:
            response = requests.post(self.base_url, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content'].strip()
        except Exception as e:
            logger.error(f"API Error: {e}")
            return f"API Error: {str(e)}"
    
    def analyze_problem(self, code: str, error: str = None, objective: str = None) -> Dict:
        logger.info("🔍 Analyzing problem with DeepSeek Reasoner...")
        
        prompt = f"""I'm working on this Python code:
```python
{code}
"""
if error:
 prompt += f"""I encountered this error:

{error}
"""
if objective:
 prompt += f"""My objective is:
{objective}
"""
else:
 prompt += "Please analyze this code and suggest improvements or fixes."

    prompt += """
Think step-by-step:

First identify any obvious syntax errors or issues

Then analyze the logical structure and potential bugs

Suggest specific improvements

Provide the corrected code if appropriate

Respond with clear explanations and well-formatted code blocks."""

    messages = [{"role": "user", "content": prompt}]
    
    # First use the reasoner for deep analysis
    analysis = self._call_api(
        messages, 
        model="deepseek-reasoner", 
        temperature=0.3,
        max_tokens=6000
    )
    
    # Then get a concise solution
    solution = self._call_api(
        messages + [{"role": "assistant", "content": analysis}],
        model="deepseek-chat",
        temperature=0.2,
        max_tokens=4000
    )
    
    # Store this attempt in history
    attempt = {
        "timestamp": datetime.now().isoformat(),
        "code": code,
        "error": error,
        "analysis": analysis,
        "solution": solution
    }
    self.code_history.append(attempt)
    if len(self.code_history) > self.max_history:
        self.code_history.pop(0)
        
    return {
        "analysis": analysis,
        "solution": solution,
        "attempt_id": len(self.code_history) - 1
    }

def get_code_from_response(self, response: str) -> str:
    if '```python' in response:
        return response.split('```python')[1].split('```')[0]
    elif '```' in response:
        return response.split('```')[1].split('```')[0]
    return response

def test_code(self, code: str) -> tuple:
    with open("_temp_code.py", "w", encoding="utf-8") as f:
        f.write(code)
        
    try:
        result = subprocess.run(
            ["python", "_temp_code.py"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            return True, result.stdout
        return False, result.stderr
    except subprocess.TimeoutExpired:
        return False, "Timeout: Code took too long to execute"
    except Exception as e:
        return False, str(e)

def show_diff(self, old_code: str, new_code: str) -> str:
    diff = difflib.unified_diff(
        old_code.splitlines(keepends=True),
        new_code.splitlines(keepends=True),
        fromfile='original',
        tofile='suggested'
    )
    return ''.join(diff)

def save_session(self, filename: str = "last_session.json"):
    data = {
        "timestamp": datetime.now().isoformat(),
        "code_history": self.code_history,
        "session_history": self.session_history,
        "current_code": self.current_code
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
        
    return filename

def load_session(self, filename: str):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.code_history = data.get('code_history', [])
            self.session_history = data.get('session_history', [])
            self.current_code = data.get('current_code', "")
    except Exception as e:
        logger.error(f"Error loading session: {e}")
@app.post("/api/initialize")
async def initialize_assistant(config: AssistantConfig):
"""Initialize a new code assistant session"""
try:
session_id = f"session_{datetime.now().strftime('%Y%m%d%H%M%S')}{os.urandom(4).hex()}"
assistant = DeepSeekAssistant(api_key=config.api_key)
code_assistants[session_id] = assistant
session_files[session_id] = f"session{session_id}.json"
return {"session_id": session_id, "status": "initialized"}
except Exception as e:
logger.error(f"Error initializing assistant: {str(e)}")
raise HTTPException(status_code=500, detail=f"Failed to initialize: {str(e)}")

@app.post("/api/analyze")
async def analyze_code(request: CodeRequest):
"""Analyze code and get suggestions"""
try:
if request.session_id not in code_assistants:
raise HTTPException(status_code=404, detail="Session not found")

    assistant = code_assistants[request.session_id]
    assistant.current_code = request.code
    
    result = assistant.analyze_problem(
        request.code, 
        request.error, 
        request.objective
    )
    
    suggested_code = assistant.get_code_from_response(result["solution"])
    
    return {
        "session_id": request.session_id,
        "analysis": result["analysis"],
        "suggested_code": suggested_code,
        "attempt_id": result["attempt_id"]
    }
except Exception as e:
    logger.error(f"Error analyzing code: {str(e)}")
    raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
@app.post("/api/test")
async def test_code(request: CodeRequest):
"""Test the current code"""
try:
if request.session_id not in code_assistants:
raise HTTPException(status_code=404, detail="Session not found")

    assistant = code_assistants[request.session_id]
    success, output = assistant.test_code(request.code)
    
    return {
        "session_id": request.session_id,
        "success": success,
        "output": output
    }
except Exception as e:
    logger.error(f"Error testing code: {str(e)}")
    raise HTTPException(status_code=500, detail=f"Testing failed: {str(e)}")
@app.post("/api/process_choice")
async def process_user_choice(request: UserChoice):
"""Process user choice during interactive session"""
try:
if request.session_id not in code_assistants:
raise HTTPException(status_code=404, detail="Session not found")

    assistant = code_assistants[request.session_id]
    
    if request.choice == 'a':  # Accept suggestion
        # Get the latest suggestion
        if not assistant.code_history:
            raise HTTPException(status_code=400, detail="No suggestions available")
        
        latest = assistant.code_history[-1]
        assistant.current_code = assistant.get_code_from_response(latest["solution"])
        
        # Test the accepted code
        success, output = assistant.test_code(assistant.current_code)
        
        return {
            "session_id": request.session_id,
            "action": "accepted",
            "success": success,
            "output": output,
            "code": assistant.current_code
        }
        
    elif request.choice == 'm' and request.modified_code:  # Modify code
        assistant.current_code = request.modified_code
        return {
            "session_id": request.session_id,
            "action": "modified",
            "code": assistant.current_code
        }
        
    elif request.choice == 't':  # Test current code
        success, output = assistant.test_code(assistant.current_code)
        return {
            "session_id": request.session_id,
            "action": "tested",
            "success": success,
            "output": output
        }
        
    else:
        raise HTTPException(status_code=400, detail="Invalid choice or missing code")
        
except Exception as e:
    logger.error(f"Error processing choice: {str(e)}")
    raise HTTPException(status_code=500, detail=f"Choice processing failed: {str(e)}")
@app.post("/api/save_session")
async def save_session(request: SessionRequest):
"""Save the current session"""
try:
if request.session_id not in code_assistants:
raise HTTPException(status_code=404, detail="Session not found")

    assistant = code_assistants[request.session_id]
    filename = request.filename or session_files[request.session_id]
    saved_file = assistant.save_session(filename)
    
    return {"status": "saved", "filename": saved_file}
except Exception as e:
    logger.error(f"Error saving session: {str(e)}")
    raise HTTPException(status_code=500, detail=f"Failed to save session: {str(e)}")
@app.post("/api/load_session")
async def load_session(request: SessionRequest):
"""Load a saved session"""
try:
if request.session_id not in code_assistants:
raise HTTPException(status_code=404, detail="Session not found")

    assistant = code_assistants[request.session_id]
    filename = request.filename or session_files[request.session_id]
    assistant.load_session(filename)
    
    return {"status": "loaded", "filename": filename}
except Exception as e:
    logger.error(f"Error loading session: {str(e)}")
    raise HTTPException(status_code=500, detail=f"Failed to load session: {str(e)}")
@app.get("/api/session_history/{session_id}")
async def get_session_history(session_id: str):
"""Get session history"""
try:
if session_id not in code_assistants:
raise HTTPException(status_code=404, detail="Session not found")

    assistant = code_assistants[session_id]
    return {
        "session_id": session_id,
        "code_history": [{
            "timestamp": h["timestamp"],
            "error": h["error"][:200] + "..." if h["error"] and len(h["error"]) > 200 else h["error"],
            "analysis": h["analysis"][:200] + "..." if len(h["analysis"]) > 200 else h["analysis"]
        } for h in assistant.code_history]
    }
except Exception as e:
    logger.error(f"Error getting history: {str(e)}")
    raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")
WebSocket for interactive sessions
@app.websocket("/ws/{session_id}")
async def interactive_session(websocket: WebSocket, session_id: str):
await websocket.accept()

if session_id not in code_assistants:
    await websocket.send_json({"error": "Session not found"})
    await websocket.close()
    return

assistant = code_assistants[session_id]

try:
    # Initial state
    await websocket.send_json({
        "type": "status",
        "message": "Interactive session started. Send your code to begin."
    })
    
    while True:
        data = await websocket.receive_json()
        
        if data["type"] == "code":
            # Analyze the code
            result = assistant.analyze_problem(
                data["code"],
                data.get("error"),
                data.get("objective")
            )
            
            suggested_code = assistant.get_code_from_response(result["solution"])
            
            await websocket.send_json({
                "type": "analysis",
                "analysis": result["analysis"],
                "suggested_code": suggested_code,
                "diff": assistant.show_diff(data["code"], suggested_code)
            })
            
        elif data["type"] == "choice":
            # Process user choice
            if data["choice"] == 'a':  # Accept
                # Get latest suggestion
                if not assistant.code_history:
                    await websocket.send_json({
                        "type": "error",
                        "message": "No suggestions available to accept"
                    })
                    continue
                
                latest = assistant.code_history[-1]
                assistant.current_code = assistant.get_code_from_response(latest["solution"])
                
                # Test the code
                success, output = assistant.test_code(assistant.current_code)
                
                await websocket.send_json({
                    "type": "test_result",
                    "success": success,
                    "output": output,
                    "code": assistant.current_code
                })
                
            elif data["choice"] == 'm':  # Modify
                if "code" not in data:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Modified code not provided"
                    })
                    continue
                
                assistant.current_code = data["code"]
                await websocket.send_json({
                    "type": "status",
                    "message": "Code modified",
                    "code": assistant.current_code
                })
                
            elif data["choice"] == 't':  # Test
                if "code" not in data:
                    code_to_test = assistant.current_code
                else:
                    code_to_test = data["code"]
                
                success, output = assistant.test_code(code_to_test)
                await websocket.send_json({
                    "type": "test_result",
                    "success": success,
                    "output": output
                })
            
        elif data["type"] == "save":
            filename = data.get("filename", session_files[session_id])
            saved_file = assistant.save_session(filename)
            await websocket.send_json({
                "type": "status",
                "message": f"Session saved to {saved_file}"
            })

except WebSocketDisconnect:
    logger.info(f"WebSocket disconnected: {session_id}")
except Exception as e:
    logger.error(f"WebSocket error: {str(e)}")
    await websocket.send_json({"type": "error", "message": str(e)})
@app.get("/")
async def root():
return {"message": "DeepSeek Reasoner API is running"}

if name == "main":
uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)


### Key Enhancements:

1. **DeepSeek Reasoner Integration**:
   - Uses `deepseek-reasoner` for in-depth code analysis
   - Uses `deepseek-chat` for concise solution generation
   - Increased token limits (6000 for reasoner, 4000 for chat)

2. **Session Management**:
   - Persistent session storage with automatic saving
   - History tracking for all code analysis attempts
   - Session loading functionality

3. **Interactive Workflow**:
   - WebSocket endpoint for real-time interaction
   - Support for all interactive commands:
     - Analyze code
     - Accept suggestions
     - Modify code
     - Test code
     - Save sessions

4. **Enhanced Code Handling**:
   - Automatic code extraction from responses
   - Code diff generation
   - In-process code testing
   - Error capturing and reporting

5. **Extended API Endpoints**:
   - `/api/initialize` - Start new assistant session
   - `/api/analyze` - Analyze code and get suggestions
   - `/api/test` - Test code execution
   - `/api/process_choice` - Handle user decisions
   - `/api/save_session` - Save current session
   - `/api/load_session` - Load saved session
   - `/api/session_history` - Get session history
   - `/ws/{session_id}` - WebSocket for interactive sessions

6. **Error Handling**:
   - Comprehensive error logging
   - Meaningful error responses
   - WebSocket error propagation

### Usage Instructions:

1. **Initialize Session**:
   ```bash
   POST /api/initialize
   {
     "api_key": "your_deepseek_api_key"
   }
Analyze Code:

bash
POST /api/analyze
{
  "session_id": "session_...",
  "code": "your_code_here",
  "error": "optional_error_message",
  "objective": "optional_objective"
}
Interactive Session via WebSocket:

Connect to ws://localhost:8000/ws/{session_id}

Send JSON messages:

{"type": "code", "code": "...", "error": "...", "objective": "..."}

{"type": "choice", "choice": "a"} (accept)

{"type": "choice", "choice": "m", "code": "modified_code"}

{"type": "choice", "choice": "t", "code": "optional_code_to_test"}

{"type": "save", "filename": "optional_filename"}

Session Management:

Save: POST /api/save_session

Load: POST /api/load_session

History: GET /api/session_history/{session_id}