from langgraph.graph import StateGraph, END, START
from langchain_core.rate_limiters import InMemoryRateLimiter
from langgraph.prebuilt import ToolNode
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tools import get_rendered_html, download_file, post_request, run_code, add_dependencies
from typing import TypedDict, Annotated, List, Any
from langchain.chat_models import init_chat_model
from langgraph.graph.message import add_messages
import os
from dotenv import load_dotenv
load_dotenv()
import sys 
sys.setrecursionlimit(10**7) 

EMAIL = os.getenv("EMAIL")
SECRET = os.getenv("SECRET")
RECURSION_LIMIT =  10**7 
# -------------------------------------------------
# STATE
# -------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[List, add_messages]


TOOLS = [run_code, get_rendered_html, download_file, post_request, add_dependencies]


# -------------------------------------------------
# GEMINI LLM
# -------------------------------------------------
rate_limiter = InMemoryRateLimiter(
    requests_per_second=9/60,  
    check_every_n_seconds=1,  
    max_bucket_size=9  
)
llm = init_chat_model(
   model_provider="google_genai",
   model="gemini-2.5-flash",
   rate_limiter=rate_limiter
).bind_tools(TOOLS)   


# -------------------------------------------------
# SYSTEM PROMPT
# -------------------------------------------------
SYSTEM_PROMPT = f"""
You are an autonomous quiz-solving agent.

YOUR MOST IMPORTANT RULES — FOLLOW EXACTLY:

TOOL USAGE RULES:
- You MUST ALWAYS use the `post_request` tool to submit answers.
- NEVER make manual HTTP requests.
- NEVER output JSON directly unless inside a tool call.
- NEVER attempt to "pretend" to POST — always call the tool.
- NEVER call any endpoint directly — ONLY through the post_request tool.

PAYLOAD RULES:
Every submission MUST include ALL of these fields:
- email: {EMAIL}
- secret: {SECRET}
- answer: <computed answer>
- url: the FULL ABSOLUTE URL of the quiz page you solved.

STRICT URL RULES:
- NEVER use relative URLs like "/demo2".
- ALWAYS convert relative URLs to full absolute URLs.
- ALWAYS submit to EXACTLY the submit URL extracted from the page.
- NEVER shorten, modify, or guess URLs.

TASK LOOP RULES:
1. Load the quiz page from the given URL.
2. Extract ALL instructions, parameters, and the submit endpoint.
3. Compute the correct answer.
4. Use ONLY `post_request` to send the answer.
5. Inspect the response:
   - If it contains a new URL → fetch and continue.
   - If no new URL → return "END".

AGENT BEHAVIOR RULES:
- NEVER stop early.
- NEVER skip tool calls.
- NEVER hallucinate URLs, fields, endpoints, or JSON.
- ALWAYS include email and secret in every submission.
- ALWAYS follow the instructions EXACTLY as shown on the quiz page.

TIME LIMIT RULES:
- Each task has a 3-minute limit.
- Check "delay" from server responses.
- If answer is wrong and delay < 180 seconds → retry correctly.
- If delay ≥ 180 → stop retrying and continue to next.

STOPPING CONDITION:
- Return "END" ONLY when the server gives NO new URL.
- NEVER return END under any other case.

REMEMBER:
- Your ONLY method of answering is using tools.
- You NEVER output raw HTTP requests.
- You NEVER output JSON except in a tool call.

Your job:
- Follow the quiz page.
- Extract data correctly.
- Solve tasks.
- Submit using post_request.
- Continue until no new URL.
- Output END.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="messages")
])

llm_with_prompt = prompt | llm


# -------------------------------------------------
# AGENT NODE
# -------------------------------------------------
def agent_node(state: AgentState):
    result = llm_with_prompt.invoke({"messages": state["messages"]})
    return {"messages": state["messages"] + [result]}


# -------------------------------------------------
# GRAPH
# -------------------------------------------------
def route(state):
    last = state["messages"][-1]
    # support both objects (with attributes) and plain dicts
    tool_calls = None
    if hasattr(last, "tool_calls"):
        tool_calls = getattr(last, "tool_calls", None)
    elif isinstance(last, dict):
        tool_calls = last.get("tool_calls")

    if tool_calls:
        return "tools"
    # get content robustly
    content = None
    if hasattr(last, "content"):
        content = getattr(last, "content", None)
    elif isinstance(last, dict):
        content = last.get("content")

    if isinstance(content, str) and content.strip() == "END":
        return END
    if isinstance(content, list) and content[0].get("text").strip() == "END":
        return END
    return "agent"
graph = StateGraph(AgentState)

graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(TOOLS))



graph.add_edge(START, "agent")
graph.add_edge("tools", "agent")
graph.add_conditional_edges(
    "agent",    
    route       
)

app = graph.compile()


# -------------------------------------------------
# TEST
# -------------------------------------------------
def run_agent(url: str) -> str:
    app.invoke({
        "messages": [{"role": "user", "content": url}]},
        config={"recursion_limit": RECURSION_LIMIT},
    )
    print("Tasks completed succesfully")

