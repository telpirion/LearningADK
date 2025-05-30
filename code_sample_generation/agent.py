import asyncio
from dotenv import load_dotenv
import json
import os
import pathlib
import textwrap

from google.adk.agents import Agent, SequentialAgent
from google.adk.events import Event
from google.adk.sessions import InMemorySessionService
from google.adk.tools import ToolContext
from google.adk.runners import Runner
from google.genai import types  # For creating message Content/Parts

# This is necessary to set the
# GOOGLE_GENAI_USE_VERTEXAI, GOOGLE_CLOUD_PROJECT, and GOOGLE_CLOUD_LOCATION
# variables
load_dotenv()

MODEL_GEMINI_2_0_FLASH = "gemini-2.0-flash"
APP_NAME = "CodeGenerator"


def get_protos(tool_context: ToolContext) -> str:
    """Gets protos from 'GitHub' for use as RAG context.

    In a real application, this would download the protos fromGitHub.

    Arguments:
        tool_context: the context passed into this tool

    Returns:
        A string containing the entire proto file
    """
    return (
        pathlib.Path(
            f"{os.path.dirname(__file__)}/resources/secretmanager.proto"
        )
        .open()
        .read()
    )


def get_evaluation(code_sample: str, tool_context: ToolContext):
    """Evaluates a code sample.

    In a real application, this tool would have a complete set of grading
    criteria.

    Arguments:
        code_sample: the code sample to evaluate
        tool_context: the context passed to this tool

    Returns:
        An evaluation as a JSON_formatted string

    """
    yield json.dumps(
        {
            "score": 1.0,
            "explanation": "This code sample is great! No notes.",
        }
    )


def init_rag_agent() -> Agent:
    """Instantiates the RAG agent.

    Returns:
        An Agent object
    """
    instruction = textwrap.dedent("""
            You are the retrieval-augmented grounding agent.
            Your task is to download protocol buffer files from GitHub
            Use the 'say_hello' tool to generate the greeting.
            If the user provides their name, make sure to pass it to the tool.
            "Do not engage in any other conversation or tasks.""")
    description = (
        "Downloads protocol buffer files from GitHub using the 'get_protos'"
        "tool."
    )
    try:
        rag_agent = Agent(
            model=MODEL_GEMINI_2_0_FLASH,
            name="rag_agent",
            instruction=textwrap.dedent(instruction),
            description=textwrap.dedent(description),
            tools=[get_protos],
        )
        print(
            f"✅ {rag_agent.name}' created using model '{rag_agent.model}'."
        )
        return rag_agent
    except Exception as e:
        print(f"❌ error: create RAG agent with ({rag_agent.model}): {e}")


def init_evaluation_agent():
    """Instantiates the evaluation agent.

    Returns:
        An Agent object
    """
    instruction = textwrap.dedent("""
        You are the Evaluation Agent. Your task is to decide how
        good a code sample is. Use the 'get_evaluation' tool when a code sample
        has been generated.""")
    description = (
        "Evaluates generated code samples using the 'get_evaluation' tool."
    )
    try:
        evaluation_agent = Agent(
            model=MODEL_GEMINI_2_0_FLASH,
            name="evaluation_agent",
            instruction=textwrap.dedent(instruction),
            description=textwrap.dedent(description),
            tools=[get_evaluation],
        )
        print(
            f"✅ Agent '{evaluation_agent.name}' created using model"
            f"'{evaluation_agent.model}'."
        )
        return evaluation_agent
    except Exception as e:
        print(
            f"❌ Could not create agent. ({evaluation_agent.model}). Error: {e}"
        )


def init_generation_agent():
    """Instantiates the generation agent.

    Returns:
        An Agent object
    """
    instruction = textwrap.dedent("""You are the Generation Agent. Your task is
                                  to write a code sample in Node.js""")
    description = "Generates code samples in Node.js"
    try:
        generation_agent = Agent(
            model=MODEL_GEMINI_2_0_FLASH,
            name="generation_agent",
            instruction=textwrap.dedent(instruction),
            description=textwrap.dedent(description),
        )
        print(
            f"✅ Agent '{generation_agent.name}' created using model"
            f"'{generation_agent.model}'."
        )
        return generation_agent
    except Exception as e:
        print(
            f"❌ Could not create agent. ({generation_agent.model}). Error: {e}"
        )


async def call_agent_async(query: str, runner, user_id, session_id):
    """Sends a query to the agent and prints the final response."""
    print(f"\n>>> User Query: {query}")

    content = types.Content(role="user", parts=[types.Part(text=query)])

    final_response_text = "Agent did not produce a final response."  # Default

    async for event in runner.run_async(
        user_id=user_id, session_id=session_id, new_message=content
    ):
        print(f"  [Event] Author: {event.author}, Type: {type(event).__name__}, Final: {event.is_final_response()}, Content: {event.content}")

        if event.is_final_response():
            if event.content and event.content.parts:
                final_response_text = event.content.parts[0].text
            elif (
                event.actions and event.actions.escalate
            ):
                final_response_text = f"Agent escalated: {event.error_message or 'No specific message.'}"
            break

    print(f"<<< Agent Response: {final_response_text}")


async def run(query):

    rag_agent = init_rag_agent()
    evaluation_agent = init_evaluation_agent()
    generation_agent = init_generation_agent()

    description = textwrap.dedent("""Executes a sequence of getting source
    grounding files, writing code samples, and evaluating the code samples.""")
    code_pipeline_agent = SequentialAgent(
        name="CodePipelineAgent",
        sub_agents=[rag_agent, evaluation_agent, generation_agent],
        description=description,
    )

    session_service = InMemorySessionService()
    session = session_service.create_session(
        app_name="CodeGenerator",
        user_id="CodeGeneratorUser",
        session_id="CodeGeneratorSession",
        state={"query": query},
    )
    session_service.append_event(
        session, Event(author="user", content={"parts": [{"text": query}]})
    )
    # Or use InMemoryRunner
    runner_agent = Runner(
        agent=code_pipeline_agent,
        app_name=APP_NAME,
        session_service=session_service,
    )

    await call_agent_async(
        query=query,
        runner=runner_agent,
        user_id="CodeGeneratorUser",
        session_id="CodeGeneratorSession",
    )


if __name__ == "__main__":
    query = textwrap.dedent("""
    Write a code sample in Node.js that gets a secret from Google Cloud Secret
    Manager. Use protos from GitHub for grounding. Be sure to evaluate the
    quality of the code sample before returning a response. Tell me what the
    evaluation of the code sample is.
    """)
    asyncio.run(run(query=query))
