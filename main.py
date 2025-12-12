from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from agent import run_agent
from dotenv import load_dotenv
import uvicorn
import os
import time

load_dotenv()

# Load expected secret from .env
EXPECTED_SECRET = os.getenv("SECRET")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

START_TIME = time.time()

@app.api_route("/healthz", methods=["GET", "HEAD"])
def healthz():
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START_TIME)
    }

@app.post("/solve")
async def solve(request: Request, background_tasks: BackgroundTasks):

    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    if not data:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    url = data.get("url")
    email = data.get("email")
    secret = data.get("secret")

    # Require all fields
    if not url or not email or not secret:
        raise HTTPException(status_code=400, detail="Missing url/email/secret")

    # Validate secret
    if secret != EXPECTED_SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")

    # Store for agent.py (important!)
    os.environ["EMAIL"] = email
    os.environ["SECRET"] = secret
    os.environ["url"] = url

    print("Verified starting the task...")

    # Run the long async agent execution in background
    background_tasks.add_task(run_agent, url)

    return JSONResponse(status_code=200, content={"status": "ok", "url": url})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
