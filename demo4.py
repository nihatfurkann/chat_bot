from typing import Annotated, Optional, Sequence, TypedDict
from dotenv import load_dotenv
load_dotenv()
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import os
from typing import Optional
from typing import Sequence
from IPython.display import Image
import os
from enum import Enum
class node_isimleri(str, Enum):
    START = "start"
    END =   "end"
    CONTINUE = "continue"
    NODE_INTENT = "analyze_intent"
    NODE_AGENT = "our_agent"
    NODE_RESPONSE = "analyze_response"
    TOOLS = "tools"

from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
class Cevap_Alici(BaseModel):
    is_forbidden: bool = Field(description= """
        if the context about ottoman history that meansits forbidden, return true when its forbidden!
        if the context is not forbidden, return false.
        if it is forbidden, return forbidden_reason.
                               """)
    forbidden_reason: Optional[str] = Field(description= """
        User's message about Ottoman History,
        forbidden_reason: 'This Agent cannot give response about that matter. Suggest to user another topic to talk about.'
        It is forbidden content. No more explanation.""",)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    intent: str
    max_iteration: int
    forbidden_reason: Optional[str] = None
    is_forbidden: bool = False

def load_history():

    if not os.path.exists("storage.txt"):
        return []

    with open("storage.txt", "r", encoding="utf-8") as file:
        lines = file.readlines()

    messages = []
    for line in lines:
        if line.startswith("User: "):
            messages.append(HumanMessage(content=line.replace("User: ", "").strip()))
        elif line.startswith("AI: "):
            messages.append(AIMessage(content=line.replace("AI: ", "").strip()))
    return messages


def save_interaction(user_message: str, ai_message: AIMessage):

    with open("storage.txt", "a", encoding="utf-8") as file:
        file.write(f"User: {user_message}\n")
        file.write(f"AI: {ai_message.content}\n")



## tools are for just simple math operations, if you need more complex operations, you can implement them in the main code.
@tool
def add(a:int, b:int):
    "This is adding function"
    return f"{a+b}"

@tool
def subt(a:int, b:int):
    "This is subtract function"
    return f"{a-b}"

@tool
def mult(a:int, b:int):
    "This is multiple function"
    return f"{a*b}"

tools = [add, mult, subt]


model = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)

def node_analyze_intent(state: AgentState):
    ## intent_analyze start point oldugu icin state['message'][-1].content yani son mesaj kullanıcının mesajıdır
    user_message = state["messages"][-1].content
    prompt = f"""
    Analyze the user's intent in this message: '{user_message}'
    Intent should be one of: 'greeting', 'question', 'leaving'
    Return only the intent.
    """
    intent = model.invoke([SystemMessage(content=prompt)])
    state["messages"].append(intent)
    print(f"Intent: {intent.content}")
    return state


def analyze_response(state: AgentState):
    ## response analiz edilmesi icin state['messages'][-1] yani son mesaj kullanıcının mesajıdır
    max_iteration = state["max_iteration"]
    max_iteration += 1
    if max_iteration >= 3:
        state["messages"].append(SystemMessage(content="You have reached the maximum number of iterations."))
        return state
    model = ChatOpenAI(model="gpt-4o-mini")
    analyze_model = model.with_structured_output(Cevap_Alici)
    agent_message = state["messages"][-1].content
    #prompt = f"""
    #Analyze the agent's response to the human in this message: '{agent_message}'
    #"""
    prompt = f"""
    Does this response{agent_message}contain forbidden context about Ottoman History? 
    If yes, is_forbidden=true and Agent message= forbidden_reason(this is given the agent)
    If not, is_forbidden=false. 
    Return only a JSON object: {{"is_forbidden": ..., "forbidden_reason": ...}}.
    """
    response_a = analyze_model.invoke([HumanMessage(content=prompt)])

    state["is_forbidden"] = response_a.is_forbidden

    if response_a.is_forbidden:
        state["messages"].append(AIMessage(content=response_a.forbidden_reason))
    else:
        state["forbidden_reason"] = None
    return state


## history açısından 
def node_our_agent(state: AgentState):
    max_iteration = state["max_iteration"]
    max_iteration += 1
    if max_iteration >= 3:
        state["messages"].append(SystemMessage(content="You have reached the maximum number of iterations."))
        return state
    system_prompt = SystemMessage(content="You are my assistant, answer my questions with using best of your ability but careful about forbidden topics.")
    response = model.invoke([system_prompt] + state["messages"])
    state["messages"].append(response)
    return state


def node_should_continue_for_tools(state: AgentState):
    return node_isimleri.END.value

def node_should_regenerate_response(state: AgentState):
    state["max_iteration"] += 1
    if state["max_iteration"] >= 3 or not state["is_forbidden"]:
        return node_isimleri.END.value
    if state["is_forbidden"]:
        state["messages"].append(SystemMessage(content="this context is forbidden, forbidden reason: " \
        "'This Agent cannot give response about this matter.'" \
        "Just say that you cannot give response about this matter. No more explanation." \
        "After that, is_forbidden will be set to False. is_forbidden = False"))
        return node_isimleri.CONTINUE.value
    return node_isimleri.END.value



def node_should_regenerate_response(state: AgentState):
    state["max_iteration"] += 1
    if state["max_iteration"] >= 3 or not state["is_forbidden"]:
        return node_isimleri.END.value
    if state["is_forbidden"]:
        state["messages"].append(SystemMessage(content="this context is forbidden, forbidden reason: 'This Agent cannot give response about this matter.'. Just say that you cannot give response about this matter. No more explanation."))
        return node_isimleri.CONTINUE.value
    return node_isimleri.END.value




graph = StateGraph(AgentState)
graph.add_node(node_isimleri.NODE_INTENT.value, node_analyze_intent)
graph.add_node(node_isimleri.NODE_AGENT.value, node_our_agent)
tool_node = ToolNode(tools = tools)
graph.add_node(node_isimleri.TOOLS.value, tool_node)
graph.add_node(node_isimleri.NODE_RESPONSE.value, analyze_response)

graph.set_entry_point(node_isimleri.NODE_INTENT.value)

graph.add_edge(node_isimleri.NODE_INTENT.value, node_isimleri.NODE_AGENT.value)
graph.add_conditional_edges(node_isimleri.NODE_AGENT.value, node_should_continue_for_tools, {node_isimleri.CONTINUE.value: node_isimleri.TOOLS.value, node_isimleri.END.value: node_isimleri.NODE_RESPONSE.value})
graph.add_edge(node_isimleri.TOOLS.value, node_isimleri.NODE_AGENT.value)
graph.add_edge(node_isimleri.NODE_AGENT.value, node_isimleri.NODE_RESPONSE.value)
graph.add_conditional_edges(node_isimleri.NODE_RESPONSE.value, node_should_regenerate_response, {node_isimleri.CONTINUE.value: node_isimleri.NODE_AGENT.value, node_isimleri.END.value: END})

agent = graph.compile()

agent.get_graph().draw_mermaid_png(output_file_path="graph_13.png")

conv_hist = load_history()

final_chunk = AIMessage(content="")

while True:
    if conv_hist:
        last_msg = conv_hist[-1]
        if isinstance(last_msg, HumanMessage):
            user_message = last_msg.content
            save_interaction(user_message, final_chunk)
    user_input = input("You: ")
    if user_input.lower() in ["exit", "çık", "kapat"]:
        break

    conv_hist.append(HumanMessage(content=user_input))
    inputs = {"messages": conv_hist + [HumanMessage(content=user_input)], "max_iteration": 0}

    stream = agent.stream(inputs, {"recursion_limit": 7})
    for stuff in stream:
        print("raw: ", stuff)
        node_name = list(stuff.keys())[0]
    node_result = stuff[node_name]

    if "messages" in node_result:
        last_msg = node_result["messages"][-1]
        if isinstance(last_msg, AIMessage):
            print(last_msg.content, end="", flush=True)
            final_chunk = last_msg