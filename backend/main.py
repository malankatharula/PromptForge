from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
import json
import os
from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="PromptForge API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

ANALYZE_SYSTEM = """You are an expert prompt engineer and LLM evaluator.
Analyze the given prompt and return ONLY valid JSON — no markdown, no explanation outside JSON.

Return this exact structure:
{
  "scores": {
    "clarity": {"score": 0-10, "reason": "one sentence"},
    "specificity": {"score": 0-10, "reason": "one sentence"},
    "context": {"score": 0-10, "reason": "one sentence"},
    "instruction_quality": {"score": 0-10, "reason": "one sentence"},
    "output_format": {"score": 0-10, "reason": "one sentence"}
  },
  "overall": 0-10,
  "verdict": "one sentence summary of what's wrong",
  "rewrites": {
    "minimal": {"prompt": "...", "what_changed": "one sentence"},
    "structured": {"prompt": "...", "what_changed": "one sentence"},
    "expert": {"prompt": "...", "what_changed": "one sentence"}
  },
  "top_issues": ["issue1", "issue2", "issue3"]
}"""

COMPARE_SYSTEM = """You are a helpful AI assistant. Answer the prompt directly and concisely."""

class AnalyzeRequest(BaseModel):
    prompt: str

class CompareRequest(BaseModel):
    original: str
    rewritten: str

@app.post("/analyze")
async def analyze_prompt(req: AnalyzeRequest):
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": ANALYZE_SYSTEM},
            {"role": "user", "content": f"Analyze this prompt:\n\n{req.prompt}"}
        ],
        temperature=0.3,
        max_tokens=2000,
    )
    raw = response.choices[0].message.content.strip()
    # strip markdown fences if model wraps in ```json
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())

@app.post("/compare")
async def compare_prompts(req: CompareRequest):
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    def run(prompt):
        r = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": COMPARE_SYSTEM},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500,
        )
        return r.choices[0].message.content.strip()

    return {
        "original_output": run(req.original),
        "rewritten_output": run(req.rewritten)
    }

@app.get("/health")
async def health():
    return {"status": "ok"}
