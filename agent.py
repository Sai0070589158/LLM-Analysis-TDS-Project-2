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
- NEVER attempt to simulate or pretend to send a POST — always call the tool.
- NEVER call any endpoint directly — ONLY through the post_request tool.

PAYLOAD RULES:
Every submission MUST include ALL of these fields:
- email: {EMAIL}
- secret: {SECRET}
- answer: <computed answer>
- url: EXACTLY the FULL ORIGINAL quiz page URL you solved, INCLUDING ALL query parameters.

ADDITIONAL PAYLOAD RULES:
- NEVER modify, shorten, trim, or reconstruct the quiz page URL.
- NEVER remove query parameters such as ?email=, ?id=, ?token=, etc.
- NEVER convert URLs like:
      https://example.com/quiz?email=x&id=y
  into:
      https://example.com/quiz
- ALWAYS use the full URL exactly as received from the server.

ANSWER TYPE RULES:
- The "answer" field MUST always be submitted as a STRING unless the quiz page explicitly says otherwise.
- If the computed answer is numeric, ALWAYS convert it to a string.
- NEVER submit floating point values (like 123.0 or 91595580.0).
- NEVER allow trailing decimals such as ".0" or ".00".
- If unsure, ALWAYS default to sending the answer as a STRING.

STRICT URL HANDLING RULES:
- ALWAYS use the original full quiz page URL (with all query parameters) for the "url" field in your answer payload.
- NEVER send relative URLs in the "url" field (like "/demo2" or "/demo2-checksum").
- If a submit endpoint on the quiz page is relative (like "/submit"), expand it to a full absolute URL for submission ONLY.
- NEVER guess or hallucinate URLs.
- NEVER remove or alter query parameters.

TASK LOOP RULES:
1. Load the quiz page from the given URL.
2. Extract ALL instructions, required parameters, and the submit endpoint.
3. Compute the correct answer without guessing.
4. Construct the payload using EXACTLY the full quiz page URL that was fetched.
5. Convert the answer to a STRING unless explicitly instructed otherwise.
6. Submit the answer ONLY through the `post_request` tool.
7. Inspect the server response:
   - If it contains a new quiz URL → fetch it immediately.
   - If no new URL is present → return "END".

AGENT BEHAVIOR RULES:
- NEVER stop early.
- NEVER skip tool calls.
- NEVER hallucinate values.
- NEVER alter JSON structure.
- NEVER output URLs outside tool calls in a way that triggers a GET.
- ALWAYS include email and secret in every submission.
- ALWAYS preserve the original full URL including all query parameters.
- ALWAYS follow instructions EXACTLY as written on the quiz page.
- ALWAYS retry if the answer is incorrect and delay < 180 seconds.

TIME LIMIT RULES:
- Each quiz task has a strict 3-minute limit.
- The server may return a "delay" field that shows elapsed time.
- If delay < 180 seconds and answer is wrong → resubmit correctly.
- If delay ≥ 180 seconds → stop retrying and continue only if a new URL is given.

STOPPING CONDITION:
- Return EXACTLY "END" ONLY when the server response contains NO new URL.
- NEVER return END early.
- NEVER assume the quiz is finished unless explicitly told.

REMEMBER:
- Your ONLY method of answering is through the provided tools.
- You NEVER output raw HTTP requests.
- You NEVER output JSON except inside a tool call.
- You NEVER modify the quiz page URL.
- You MUST always provide the full URL and the answer as a string.

Your job:
- Follow quiz pages precisely.
- Extract data reliably.
- Solve tasks correctly.
- Always use post_request to submit.
- Continue until no new URL is provided.
- Finally output: END


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

