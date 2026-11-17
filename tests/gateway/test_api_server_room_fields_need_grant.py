"""POST /v1/runs refuses room dispatch fields that arrive without a room grant (TIN-8482).

``_normalize_room_dispatch`` resolves ``_room_execution_policy`` from this gateway's own catalog
for a verified ``HermesRoom`` grant. Before this check an API-key request could bring both
fields in its body, and the policy's ``enabled_toolsets`` replaced the profile's
``platform_toolsets.api_server`` for that run. A tool-free profile then ran with terminal,
file and web tools.
"""

import hashlib
import json
from unittest.mock import MagicMock, patch

import pytest
from aiohttp.test_utils import TestClient, TestServer

from gateway.hosted_rooms_common import compact_json
from gateway.platforms import api_server
from tests.gateway.test_api_server_runs import _create_runs_app, _make_adapter
from tests.waiting import async_wait_until


def _self_made_policy() -> dict:
    unsigned = {
        "version": 1, "target_profile": "default",
        "enabled_toolsets": ["bot_room", "file", "terminal", "web"],
        "approval_mode": "off", "max_iterations": 8}
    digest = hashlib.sha256(compact_json(unsigned).encode("ascii")).hexdigest()
    return {**unsigned, "policy_digest": digest}


def _refused_bodies() -> dict[str, dict]:
    policy = _self_made_policy()
    return {
        "dispatch and inline policy": {
            "input": "x", "hosted_room_dispatch": {"room_id": "synthetic-room"},
            "_room_execution_policy": policy},
        "dispatch by digest only": {
            "input": "x", "hosted_room_dispatch": {
                "room_id": "synthetic-room", "execution_policy_digest": policy["policy_digest"]}},
        "inline policy only": {"input": "x", "_room_execution_policy": policy},
        "null dispatch": {"input": "x", "hosted_room_dispatch": None},
        "empty policy": {"input": "x", "_room_execution_policy": {}},
    }


def _passed_bodies() -> dict[str, dict]:
    """Fields _handle_runs never reads as a room policy: they reach it and set no toolset."""
    policy = _self_made_policy()
    return {
        "policy nested in metadata": {
            "input": "x", "metadata": {"hosted_room_dispatch": {"room_id": "r"},
                                       "_room_execution_policy": policy}},
        "toolset aliases": {
            "input": "x", "toolsets": ["terminal"], "enabled_toolsets": ["terminal"],
            "platform_toolsets": {"api_server": ["terminal"]}, "room_execution_policy": policy},
        "connector fields": {
            "input": "x", "connectors": ["gmail"], "connector": "gmail",
            "tools": {"connectors": {"enabled": True}}},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("name", sorted(_refused_bodies()))
async def test_room_fields_without_a_grant_are_refused_before_work(name):
    adapter = api_server.APIServerAdapter.__new__(api_server.APIServerAdapter)
    adapter._room_grant_token = MagicMock(return_value="")
    body = _refused_bodies()[name]

    normalized, error = await adapter._normalize_room_dispatch(object(), body)

    assert normalized is body
    assert error is not None and error.status == 400
    assert json.loads(error.text)["error"]["code"] == "room_grant_required"


@pytest.mark.asyncio
async def test_an_api_key_run_with_room_fields_builds_no_agent():
    adapter = _make_adapter(api_key="sk-synthetic-room-fields")
    app = _create_runs_app(adapter)
    headers = {"Authorization": "Bearer sk-synthetic-room-fields"}
    with patch.object(adapter, "_create_agent") as create:
        async with TestClient(TestServer(app)) as cli:
            for name, body in _refused_bodies().items():
                response = await cli.post("/v1/runs", json=body, headers=headers)
                payload = await response.json()
                assert response.status == 400, (name, payload)
                assert payload["error"]["code"] == "room_grant_required", name
    create.assert_not_called()
    assert adapter._pending_agent_requests == 0


@pytest.mark.asyncio
async def test_other_fields_reach_the_run_with_no_room_policy():
    adapter = _make_adapter(api_key="sk-synthetic-room-fields")
    app = _create_runs_app(adapter)
    headers = {"Authorization": "Bearer sk-synthetic-room-fields"}
    with patch.object(adapter, "_create_agent") as create:
        agent = MagicMock()
        agent.run_conversation.return_value = {"final_response": "done"}
        agent.session_prompt_tokens = agent.session_completion_tokens = agent.session_total_tokens = 0
        create.return_value = agent
        async with TestClient(TestServer(app)) as cli:
            for name, body in _passed_bodies().items():
                response = await cli.post("/v1/runs", json=body, headers=headers)
                assert response.status == 202, (name, await response.json())
            await async_wait_until(
                lambda: create.call_count == len(_passed_bodies()), what="every passed body built its agent")
    for call in create.call_args_list:
        assert call.kwargs["room_dispatch"] is None
        assert call.kwargs["room_execution_policy"] is None
