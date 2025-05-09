import json
import os
import pathlib
from typing import Any, Mapping

from google.adk.agents import Agent


MODEL_GEMINI_2_0_FLASH = "gemini-2.0-flash"

root_agent_var_name = "code_sample_generation_agent_team"
root_agent = None
runner_root = None


def get_protos():
    return (
        pathlib.Path(f"{os.path.dirname(__file__)}/resources/secretmanager.proto")
        .open()
        .read()
    )


def get_evaluation(code_sample: str) -> str:
    return json.dumps({
        "score": 1.0,
        "explanation": "This code sample is great! No notes.",
    })


def generate_code_sample(grounding: str) -> str:
    assert grounding != "", "No grounding context provided"

    return (
        pathlib.Path(f"{os.path.dirname(__file__)}/resources/get-secret.js")
        .open()
        .read()
    )


# -- RAG Agent --
rag_agent = None
try:
    rag_agent = Agent(
        model=MODEL_GEMINI_2_0_FLASH,
        name="rag_agent",
        instruction="You are the retrieval-augmented grounding agent. Your ONLY task is to download protocol buffer files from GitHub"
        "Use the 'say_hello' tool to generate the greeting. "
        "If the user provides their name, make sure to pass it to the tool. "
        "Do not engage in any other conversation or tasks.",
        description="Downloads protocol buffer files from GitHub using the 'get_protos' tool.",
        tools=[get_protos],
    )
    print(
        f"✅ Agent '{rag_agent.name}' created using model '{rag_agent.model}'."
    )
except Exception as e:
    print(
        f"❌ Could not create RAG agent. Check API Key ({rag_agent.model}). Error: {e}"
    )

# --- Evaluation Agent ---
evaluation_agent = None
try:
    evaluation_agent = Agent(
        model=MODEL_GEMINI_2_0_FLASH,
        name="evaluation_agent",
        instruction="You are the Evaluation Agent. Your ONLY task is to decide how good a code sample is."
        "Use the 'get_evaluation' tool when a code sample has been generated.",
        description="Evaluates generated code samples using the 'get_evaluation' tool.",  # Crucial for delegation
        tools=[get_evaluation],
    )
    print(
        f"✅ Agent '{evaluation_agent.name}' created using model"
        f"'{evaluation_agent.model}'."
    )
except Exception as e:
    print(
        f"❌ Could not create agent. ({evaluation_agent.model}). Error: {e}"
    )


# --- Evaluation Agent ---
generation_agent = None
try:
    generation_agent = Agent(
        model=MODEL_GEMINI_2_0_FLASH,
        name="generation_agent",
        instruction="You are the Generation Agent. Your ONLY task is to write a code sample in Node.js",
        description="Generates code samples ",
        #tools=[generate_code_sample],
    )
    print(
        f"✅ Agent '{generation_agent.name}' created using model"
        f"'{generation_agent.model}'."
    )
except Exception as e:
    print(
        f"❌ Could not create agent. ({generation_agent.model}). Error: {e}"
    )

root_agent = Agent(
    name="code_sample_generation",
    model=MODEL_GEMINI_2_0_FLASH,
    description="The main coordinator agent. Handles code generation requests and delegates RAG and evaluations to specialists",
    instruction="You are the main Code Sample Generation Agent coordinating a team. Your primary responsibility is to create a Google Cloud code sample in Node.js."
    "You have specialized sub-agents: "
    "1. 'rag_agent': Handles getting protocol buffers from GitHub. Use this tool to get grounding context before generating the code sample."
    "2. 'evaluation_agent': Handles evaluating a code sample to determine quality.After generating a code sample, use this tool to evaluate it."
    "Analyze the user's query. If it is a request to generate a code sample,"
    "first call the 'rag_agent' to get the grounding context. Next, use the 'generation_agent' to generate a code sample based on the grounding context. Next, you MUST use the 'evaluation_agent' to evaluate the quality of the code sample. Finally,"
    "show the user the code sample and tell them how good it is."
    "For anything else, respond appropriately or state you cannot handle it.",
    sub_agents=[rag_agent, evaluation_agent, generation_agent],
)
