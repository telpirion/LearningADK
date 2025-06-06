from agent import (
  init_rag_agent,
  init_evaluation_agent,
  init_generation_agent
)


def test_init_rag_agent(capsys):
    agent = init_rag_agent()
    assert agent is not None
    assert agent.tools is not None
    assert "get_protos" in agent.instruction

    # TODO: execute agent

    actual_response = capsys.readouterr()
    assert actual_response.out != ""
    assert "rag_agent" in actual_response.out


def test_init_eval_agent(capsys):
    agent = init_evaluation_agent()
    assert agent is not None
    assert agent.tools is not None
    assert "get_evaluation" in agent.instruction

    # TODO: execute agent

    actual_response = capsys.readouterr()
    assert actual_response.out != ""
    assert "evaluation_agent" in actual_response.out


def test_init_generation_agent(capsys):
    agent = init_generation_agent()
    assert agent is not None
    assert "Generation Agent" in agent.instruction

    actual_response = capsys.readouterr()
    assert actual_response.out != ""
    assert "generation_agent" in actual_response.out
