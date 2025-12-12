from langgraph.graph import StateGraph, END, START
from shared_store import url_time
import time
from langchain_core.rate_limiters import InMemoryRateLimiter
from langgraph.prebuilt import ToolNode
from tools import (
    get_rendered_html, download_file, post_request,
    run_code, add_dependencies, ocr_image_tool, transcribe_audio, encode_image_to_base64
)
from typing import TypedDict, Annotated, List
from langchain_core.messages import HumanMessage
from langchain.chat_models import init_chat_model
from langgraph.graph.message import add_messages
import os
from dotenv import load_dotenv
load_dotenv()

EMAIL = os.getenv("EMAIL")
SECRET = os.getenv("SECRET")

RECURSION_LIMIT = 5000
MAX_TOKENS = 60000


# -------------------------------------------------
# STATE
# -------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[List, add_messages]


TOOLS = [
    run_code, get_rendered_html, download_file,
    post_request, add_dependencies, ocr_image_tool, transcribe_audio, encode_image_to_base64
]


# -------------------------------------------------
# LLM INIT (OPENROUTER COMPATIBLE)
# -------------------------------------------------
rate_limiter = InMemoryRateLimiter(
    requests_per_second=1,
    check_every_n_seconds=1,
    max_bucket_size=5
)

llm = init_chat_model(
    model_provider="openai",
    model="google/gemini-2.0-flash-exp",
    rate_limiter=rate_limiter,
    client_params={
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": os.getenv("OPENROUTER_API_KEY"),
        "headers": {
            "HTTP-Referer": "https://your-app.com",
            "X-Title": "TDS-Project-2-Agent"
        }
    }
).bind_tools(TOOLS)


# -------------------------------------------------
# SYSTEM PROMPT
# -------------------------------------------------
SYSTEM_PROMPT = f"""
You are an autonomous quiz-solving agent.

Your job is to:
1. Load each quiz page from the given URL.
2. Extract instructions, parameters, and submit endpoint.
3. Solve tasks exactly.
4. Submit answers ONLY to the correct endpoint.
5. Follow new URLs until none remain, then output END.

Rules:
- For base64 generation of an image NEVER use your own code, always use the "encode_image_to_base64" tool that's provided
- Never hallucinate URLs or fields.
- Never shorten endpoints.
- Always inspect server response.
- Never stop early.
- Use tools for HTML, downloading, rendering, OCR, or running code.
- Include:
    email = {EMAIL}
    secret = {SECRET}
"""


# -------------------------------------------------
# NEW NODE: HANDLE MALFORMED JSON
# -------------------------------------------------
def handle_malformed_node(state: AgentState):
    print("--- DETECTED MALFORMED JSON. ASKING AGENT TO RETRY ---")
    return {
        "messages": [
            {
                "role": "user",
                "content": "SYSTEM ERROR: Your last tool call was Malformed (Invalid JSON). Fix and retry."
            }
        ]
    }


# -------------------------------------------------
# AGENT NODE
# -------------------------------------------------
def agent_node(state: AgentState):

    # --- TIME HANDLING START ---
    cur_time = time.time()
    cur_url = os.getenv("url")

    prev_time = url_time.get(cur_url)
    offset = os.getenv("offset", "0")

    if prev_time is not None:
        prev_time = float(prev_time)
        diff = cur_time - prev_time

        if diff >= 180 or (offset != "0" and (cur_time - float(offset)) > 90):
            print(f"Timeout exceeded ({diff}s) — submitting WRONG answer")

            fail_instruction = """
            You have exceeded the time limit (180s).
            Immediately call `post_request` and submit a WRONG answer.
            """

            fail_msg = HumanMessage(content=fail_instruction)
            result = llm.invoke(state["messages"] + [fail_msg])
            return {"messages": [result]}

    # -------------------------------------------------
    # SAFE TRIMMING (NO TOKEN COUNTING)
    # -------------------------------------------------
    print("⚠ Model does not support token counting — trimming to last 40 messages")
    trimmed_messages = state["messages"]
    if len(trimmed_messages) > 40:
        trimmed_messages = trimmed_messages[-40:]

    # Ensure at least one human message exists
    has_human = any(getattr(msg, "type", None) == "human" for msg in trimmed_messages)
    if not has_human:
        current_url = os.getenv("url", "Unknown URL")
        print("WARNING: No human message found. Injecting reminder.")
        trimmed_messages.append(
            HumanMessage(content=f"Context lost. Continue processing URL: {current_url}")
        )

    print(f"--- INVOKING AGENT (Context: {len(trimmed_messages)} items) ---")
    result = llm.invoke(trimmed_messages)
    return {"messages": [result]}


# -------------------------------------------------
# ROUTE LOGIC
# -------------------------------------------------
def route(state):
    last = state["messages"][-1]

    if "finish_reason" in last.response_metadata:
        if last.response_metadata["finish_reason"] == "MALFORMED_FUNCTION_CALL":
            return "handle_malformed"

    if getattr(last, "tool_calls", None):
        print("Route → tools")
        return "tools"

    content = getattr(last, "content", None)

    if isinstance(content, str) and content.strip() == "END":
        return END

    if isinstance(content, list) and len(content) and isinstance(content[0], dict):
        if content[0].get("text", "").strip() == "END":
            return END

    print("Route → agent")
    return "agent"


# -------------------------------------------------
# GRAPH
# -------------------------------------------------
graph = StateGraph(AgentState)

graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(TOOLS))
graph.add_node("handle_malformed", handle_malformed_node)

graph.add_edge(START, "agent")
graph.add_edge("tools", "agent")
graph.add_edge("handle_malformed", "agent")

graph.add_conditional_edges(
    "agent",
    route,
    {
        "tools": "tools",
        "agent": "agent",
        "handle_malformed": "handle_malformed",
        END: END
    }
)

app = graph.compile()


# -------------------------------------------------
# RUNNER
# -------------------------------------------------
def run_agent(url: str):
    initial_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": url}
    ]

    app.invoke(
        {"messages": initial_messages},
        config={"recursion_limit": RECURSION_LIMIT}
    )

    print("Tasks completed successfully!")
