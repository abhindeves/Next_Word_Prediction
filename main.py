from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import google.generativeai as genai
from dotenv import load_dotenv
import os
import json
import re
import time
from typing import Optional

# Load environment variables
load_dotenv()

# Configure FastAPI
app = FastAPI(title="Next Word Predictor", 
              description="API for predicting next word probabilities using Gemini",
              version="1.0.0")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Configure Gemini
try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
except KeyError:
    raise RuntimeError("Please set GEMINI_API_KEY in your .env file")

# Model configuration
generation_config = {
    "temperature": 0.7,
    "top_p": 1,
    "top_k": 40,
    "max_output_tokens": 200
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config
)

# Request validation
class PredictionRequest(BaseModel):
    text: str

# Retry logic
def call_llm_with_retries(prompt, model, max_retries=5, initial_backoff=0.5, backoff_factor=2):
    backoff = initial_backoff

    for attempt in range(1, max_retries + 1):
        try:
            response = model.generate_content(prompt)
            raw_text = response.candidates[0].content.parts[0].text
            cleaned = re.sub(r"```json|```", "", raw_text).strip()
            predictions = json.loads(cleaned)

            # Convert to list format for frontend
            result = [{"word": k, "probability": v} for k, v in predictions.items()]
            return JSONResponse(content=result)

        except (json.JSONDecodeError, KeyError, IndexError) as parse_err:
            if attempt < max_retries:
                time.sleep(backoff)
                backoff *= backoff_factor
                continue
            raise HTTPException(
                status_code=500,
                detail=f"Failed to parse model response after {max_retries} attempts",
            ) from parse_err

        except Exception as llm_err:
            raise HTTPException(status_code=500, detail=str(llm_err)) from llm_err

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict_next_word(request: PredictionRequest):
    input_text = request.text

    if not input_text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty")

    prompt = f"""
    You are a probabilistic language model analyzer. For the following text:
    "{input_text}"

    Return ONLY a JSON dictionary with exactly 5 key-value pairs where:
    - Each key is a likely next word
    - Each value is its estimated probability (0-1)

    Format example:
    ```json
    {{
        "word1": 0.35,
        "word2": 0.25,
        "word3": 0.15,
        "word4": 0.10,
        "word5": 0.05
    }}
    ```

    Ensure the probabilities sum to approximately 1.0.
    """

    return call_llm_with_retries(prompt, model)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
