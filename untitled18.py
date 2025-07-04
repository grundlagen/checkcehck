
+from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
+from fastapi.middleware.cors import CORSMiddleware
+from pydantic import BaseModel
+import uvicorn
+import json
+import os
+import asyncio
+from datetime import datetime
+from typing import List, Dict, Optional
+import logging
+import subprocess
+import difflib
+import requests
+
+# Setup logging
+logging.basicConfig(level=logging.INFO)
+logger = logging.getLogger(__name__)
+
+app = FastAPI(title="DeepSeek RecThink API", description="CoRT using DeepSeek Reasoner")
+
+# CORS for development
+app.add_middleware(
+    CORSMiddleware,
+    allow_origins=["*"],
+    allow_credentials=True,
+    allow_methods=["*"],
+    allow_headers=["*"],
+)
+
+# Session storage
+code_assistants: Dict[str, "DeepSeekAssistant"] = {}
+session_files: Dict[str, str] = {}
+
+
+class AssistantConfig(BaseModel):
+    api_key: str
+
+
+class CodeRequest(BaseModel):
+    session_id: str
+    code: str
+    error: Optional[str] = None
+    objective: Optional[str] = None
+
+
+class SessionRequest(BaseModel):
+    session_id: str
+    filename: Optional[str] = None
+
+
+class UserChoice(BaseModel):
+    session_id: str
+    choice: str
+    modified_code: Optional[str] = None
+
+
+class DeepSeekAssistant:
+    """Assistant that queries the DeepSeek Reasoner API."""
+
+    def __init__(self, api_key: Optional[str] = None):
+        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
+        self.base_url = "https://api.deepseek.com/v1/chat/completions"
+        self.headers = {
+            "Authorization": f"Bearer {self.api_key}",
+            "Content-Type": "application/json",
+        }
+        self.code_history: List[Dict] = []
+        self.session_history: List[Dict] = []
+        self.max_history = 5
+        self.current_code = ""
+        self.last_error = ""
+
+    def _call_api(
+        self,
+        messages: List[Dict],
+        model: str = "deepseek-chat",
+        temperature: float = 0.7,
+        max_tokens: int = 4096,
+    ) -> str:
+        payload = {
+            "model": model,
+            "messages": messages,
+            "temperature": temperature,
+            "max_tokens": max_tokens,
+        }
+        try:
+            response = requests.post(self.base_url, headers=self.headers, json=payload)
+            response.raise_for_status()
+            data = response.json()
+            return data["choices"][0]["message"]["content"].strip()
+        except Exception as e:
+            logger.error(f"API Error: {e}")
+            return f"API Error: {e}"
+
+    def analyze_problem(self, code: str, error: str | None = None, objective: str | None = None) -> Dict:
+        logger.info("Analyzing problem with DeepSeek Reasoner")
+        prompt = f"I'm working on this Python code:\n```python\n{code}\n```\n"
+        if error:
+            prompt += f"I encountered this error:\n\n{error}\n"
+        if objective:
+            prompt += f"My objective is:\n{objective}\n"
+        else:
+            prompt += "Please analyze this code and suggest improvements or fixes."
+
+        prompt += (
+            "\nThink step-by-step:\n"
+            "- Identify syntax errors or issues\n"
+            "- Analyze the logical structure and potential bugs\n"
+            "- Suggest specific improvements\n"
+            "- Provide corrected code if appropriate\n"
+        )
+
+        messages = [{"role": "user", "content": prompt}]
+
+        analysis = self._call_api(
+            messages,
+            model="deepseek-reasoner",
+            temperature=0.3,
+            max_tokens=6000,
+        )
+        solution = self._call_api(
+            messages + [{"role": "assistant", "content": analysis}],
+            model="deepseek-chat",
+            temperature=0.2,
+            max_tokens=4000,
+        )
+
+        attempt = {
+            "timestamp": datetime.now().isoformat(),
+            "code": code,
+            "error": error,
+            "analysis": analysis,
+            "solution": solution,
+        }
+        self.code_history.append(attempt)
+        if len(self.code_history) > self.max_history:
+            self.code_history.pop(0)
+
+        return {"analysis": analysis, "solution": solution, "attempt_id": len(self.code_history) - 1}
+
+    @staticmethod
+    def get_code_from_response(response: str) -> str:
+        if "```python" in response:
+            return response.split("```python")[1].split("```")[0]
+        if "```" in response:
+            return response.split("```")[1].split("```")[0]
+        return response
+
+    def test_code(self, code: str) -> tuple[bool, str]:
+        with open("_temp_code.py", "w", encoding="utf-8") as f:
+            f.write(code)
+        try:
+            result = subprocess.run(
+                ["python", "_temp_code.py"], capture_output=True, text=True, timeout=10
+            )
+            if result.returncode == 0:
+                return True, result.stdout
+            return False, result.stderr
+        except subprocess.TimeoutExpired:
+            return False, "Timeout: Code took too long to execute"
+        except Exception as e:
+            return False, str(e)
+
+    @staticmethod
+    def show_diff(old_code: str, new_code: str) -> str:
+        diff = difflib.unified_diff(
+            old_code.splitlines(keepends=True),
+            new_code.splitlines(keepends=True),
+            fromfile="original",
+            tofile="suggested",
+        )
+        return "".join(diff)
+
+    def save_session(self, filename: str) -> str:
+        data = {
+            "timestamp": datetime.now().isoformat(),
+            "code_history": self.code_history,
+            "session_history": self.session_history,
+            "current_code": self.current_code,
+        }
+        with open(filename, "w", encoding="utf-8") as f:
+            json.dump(data, f, indent=2)
+        return filename
+
+    def load_session(self, filename: str) -> None:
+        with open(filename, "r", encoding="utf-8") as f:
+            data = json.load(f)
+            self.code_history = data.get("code_history", [])
+            self.session_history = data.get("session_history", [])
+            self.current_code = data.get("current_code", "")
+
+
+@app.post("/api/initialize")
+async def initialize_assistant(config: AssistantConfig):
+    """Initialize a new assistant session."""
+    try:
+        session_id = f"session_{datetime.now().strftime('%Y%m%d%H%M%S')}_{os.urandom(4).hex()}"
+        assistant = DeepSeekAssistant(api_key=config.api_key)
+        code_assistants[session_id] = assistant
+        session_files[session_id] = f"session_{session_id}.json"
+        return {"session_id": session_id, "status": "initialized"}
+    except Exception as e:
+        logger.error(f"Error initializing assistant: {e}")
+        raise HTTPException(status_code=500, detail=f"Failed to initialize: {e}")
+
+
+@app.post("/api/analyze")
+async def analyze_code(request: CodeRequest):
+    if request.session_id not in code_assistants:
+        raise HTTPException(status_code=404, detail="Session not found")
+
+    assistant = code_assistants[request.session_id]
+    assistant.current_code = request.code
+    result = assistant.analyze_problem(request.code, request.error, request.objective)
+    suggested = assistant.get_code_from_response(result["solution"])
+    return {
+        "session_id": request.session_id,
+        "analysis": result["analysis"],
+        "suggested_code": suggested,
+        "attempt_id": result["attempt_id"],
+    }
+
+
+@app.post("/api/test")
+async def test_current_code(request: CodeRequest):
+    if request.session_id not in code_assistants:
+        raise HTTPException(status_code=404, detail="Session not found")
+
+    assistant = code_assistants[request.session_id]
+    success, output = assistant.test_code(request.code)
+    return {"session_id": request.session_id, "success": success, "output": output}
+
+
+@app.post("/api/process_choice")
+async def process_choice(request: UserChoice):
+    if request.session_id not in code_assistants:
+        raise HTTPException(status_code=404, detail="Session not found")
+
+    assistant = code_assistants[request.session_id]
+
+    if request.choice == "a":
+        if not assistant.code_history:
+            raise HTTPException(status_code=400, detail="No suggestions available")
+        latest = assistant.code_history[-1]
+        assistant.current_code = assistant.get_code_from_response(latest["solution"])
+        success, output = assistant.test_code(assistant.current_code)
+        return {
+            "session_id": request.session_id,
+            "action": "accepted",
+            "success": success,
+            "output": output,
+            "code": assistant.current_code,
+        }
+    elif request.choice == "m" and request.modified_code:
+        assistant.current_code = request.modified_code
+        return {"session_id": request.session_id, "action": "modified", "code": assistant.current_code}
+    elif request.choice == "t":
+        success, output = assistant.test_code(assistant.current_code)
+        return {
+            "session_id": request.session_id,
+            "action": "tested",
+            "success": success,
+            "output": output,
+        }
+    else:
+        raise HTTPException(status_code=400, detail="Invalid choice or missing code")
+
+
+@app.post("/api/save_session")
+async def save_session_endpoint(request: SessionRequest):
+    if request.session_id not in code_assistants:
+        raise HTTPException(status_code=404, detail="Session not found")
+    assistant = code_assistants[request.session_id]
+    filename = request.filename or session_files[request.session_id]
+    saved = assistant.save_session(filename)
+    return {"status": "saved", "filename": saved}
+
+
+@app.post("/api/load_session")
+async def load_session_endpoint(request: SessionRequest):
+    if request.session_id not in code_assistants:
+        raise HTTPException(status_code=404, detail="Session not found")
+    assistant = code_assistants[request.session_id]
+    filename = request.filename or session_files[request.session_id]
+    assistant.load_session(filename)
+    return {"status": "loaded", "filename": filename}
+
+
+@app.get("/api/session_history/{session_id}")
+async def session_history(session_id: str):
+    if session_id not in code_assistants:
+        raise HTTPException(status_code=404, detail="Session not found")
+    assistant = code_assistants[session_id]
+    hist = []
+    for h in assistant.code_history:
+        hist.append({
+            "timestamp": h["timestamp"],
+            "error": (h["error"][:200] + "...") if h.get("error") and len(h["error"]) > 200 else h.get("error"),
+            "analysis": (h["analysis"][:200] + "...") if len(h["analysis"]) > 200 else h["analysis"],
+        })
+    return {"session_id": session_id, "code_history": hist}
+
+
+@app.websocket("/ws/{session_id}")
+async def interactive_session(websocket: WebSocket, session_id: str):
+    await websocket.accept()
+    if session_id not in code_assistants:
+        await websocket.send_json({"error": "Session not found"})
+        await websocket.close()
+        return
+
+    assistant = code_assistants[session_id]
+    try:
+        await websocket.send_json({"type": "status", "message": "Interactive session started."})
+        while True:
+            data = await websocket.receive_json()
+            if data["type"] == "code":
+                result = assistant.analyze_problem(data["code"], data.get("error"), data.get("objective"))
+                suggested = assistant.get_code_from_response(result["solution"])
+                await websocket.send_json({
+                    "type": "analysis",
+                    "analysis": result["analysis"],
+                    "suggested_code": suggested,
+                    "diff": assistant.show_diff(data["code"], suggested),
+                })
+            elif data["type"] == "choice":
+                if data["choice"] == "a":
+                    if not assistant.code_history:
+                        await websocket.send_json({"type": "error", "message": "No suggestions available"})
+                        continue
+                    latest = assistant.code_history[-1]
+                    assistant.current_code = assistant.get_code_from_response(latest["solution"])
+                    success, output = assistant.test_code(assistant.current_code)
+                    await websocket.send_json({
+                        "type": "test_result",
+                        "success": success,
+                        "output": output,
+                        "code": assistant.current_code,
+                    })
+                elif data["choice"] == "m":
+                    if "code" not in data:
+                        await websocket.send_json({"type": "error", "message": "Modified code not provided"})
+                        continue
+                    assistant.current_code = data["code"]
+                    await websocket.send_json({"type": "status", "message": "Code modified", "code": assistant.current_code})
+                elif data["choice"] == "t":
+                    code_to_test = data.get("code", assistant.current_code)
+                    success, output = assistant.test_code(code_to_test)
+                    await websocket.send_json({"type": "test_result", "success": success, "output": output})
+            elif data["type"] == "save":
+                filename = data.get("filename", session_files[session_id])
+                saved_file = assistant.save_session(filename)
+                await websocket.send_json({"type": "status", "message": f"Session saved to {saved_file}"})
+    except WebSocketDisconnect:
+        logger.info(f"WebSocket disconnected: {session_id}")
+    except Exception as e:
+        logger.error(f"WebSocket error: {e}")
+        await websocket.send_json({"type": "error", "message": str(e)})
+
+
+@app.get("/")
+async def root():
+    return {"message": "DeepSeek RecThink API is running"}
+
+
+if __name__ == "__main__":
+    uvicorn.run("deepseek_recthink:app", host="0.0.0.0", port=8000, reload=True)
