from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Literal
import pandas as pd
import numpy as np
import io
import re
import time
from uuid import uuid4
import logging

# Agent imports
try:
    from thinking_agent import stream_chat
    from groq_coding_agent import stream_chat as coding_chat
    from auto_research_agent import auto_research
except ImportError:
    raise ImportError("Ensure groq_math, groq_coding_agent, and auto_research_agent are available on PYTHONPATH")

app = FastAPI(title="Thinking Agent API")

# Regex heuristics
_REGEX_MATH = re.compile(r"\b(integrate|\u222B|derivative|limit|solve|root|dx)\b", re.I)
_REGEX_CODE = re.compile(r"\b(class|def|algorithm|complexity|python|golang|java|csharp)\b", re.I)

# In-memory session-specific DataFrame store
df_store: dict[str, pd.DataFrame] = {}

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ThinkingAgentLogger")

# Request/Response schemas
class PromptRequest(BaseModel):
    prompt: str
    session_id: Optional[str] = None
    task_type: Optional[Literal["math", "code", "research", "auto"]] = "auto"

class PromptResponse(BaseModel):
    task_type: str
    output: str | None
    explanation: str | None
    session_id: str
    model: Optional[str] = None
    confidence: Optional[float] = None
    timestamp: float

# Utility: log task interaction
def log_task(session_id: str, task_type: str, prompt: str, output: str | None, model: str, explanation: str | None):
    logger.info({
        "session_id": session_id,
        "task_type": task_type,
        "prompt": prompt,
        "output": output,
        "model": model,
        "explaination": explanation,
        "timestamp": time.time()
    })

# ─────────────────────────── Routes ───────────────────────────

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), session_id: str = Form(...)):
    try:
        if not file.filename:
            raise Exception("No file name found, something corrupted!")
        ext = file.filename.lower().split(".")[-1]
        content = await file.read()
        if ext in ["csv", "tsv"]:
            sep = "\t" if ext == "tsv" else ","
            df = pd.read_csv(io.BytesIO(content), sep=sep)
        elif ext in ["xlsx"]:
            df = pd.read_excel(io.BytesIO(content))
        else:
            return JSONResponse(status_code=400, content={"error": "Unsupported file format."})

        # Basic preprocessing: drop all-null columns, infer types
        df.dropna(axis=1, how='all', inplace=True)
        df_store[session_id] = df.infer_objects()

        return {"message": "File uploaded", "rows": len(df), "columns": df.columns.tolist()}
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/df_eval")
def dataframe_eval(expr: str = Form(...), session_id: str = Form(...)):
    df = df_store.get(session_id)
    if df is None:
        return JSONResponse(status_code=404, content={"error": "No DataFrame loaded."})
    try:
        result = eval(expr, {"__builtins__": {}}, {"df": df, "pd": pd, "np": np})
        output = result.to_string(index=False) if isinstance(result, pd.DataFrame) and result.size <= 200 else str(result)
        return {"result": output}
    except Exception as e:
        logger.error(f"DataFrame eval error: {e}")
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.post("/analyze", response_model=PromptResponse)
async def analyze_prompt(req: PromptRequest):
    session_id = req.session_id or str(uuid4())
    prompt = req.prompt.strip()
    task = req.task_type or "auto"

    output, explanation, model, task_type = "", "", "", task
    confidence = None
    timestamp = time.time()

    try:
        if task == "math" or (task == "auto" and _REGEX_MATH.search(prompt)):
            output, explanation = stream_chat(prompt)
            model = "groq-math"
            task_type = "math"
            confidence = 1.0

        elif task == "code" or (task == "auto" and _REGEX_CODE.search(prompt)):
            output, explanation = coding_chat(prompt)
            model = "groq-coding"
            task_type = "code"
            confidence = 0.8

        else:
            output, explanation = auto_research(prompt, k=3)
            model = "groq-research"
            task_type = "research"
            confidence = 0.6
    except Exception as e:
        output = f"Agent error: {str(e)}"
        explanation = "An error occurred while processing the prompt."

    log_task(session_id, task_type, prompt, output, model, explanation)

    return PromptResponse(
        task_type=task_type,
        output=output,
        explanation=explanation,
        session_id=session_id,
        model=model,
        confidence=confidence,
        timestamp=timestamp
    )
