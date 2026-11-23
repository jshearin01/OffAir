"""Manual MCP verification — real SDK client against real sessions.

Not run by pytest (like the other manual_*.py scripts). Spawns the
OffAir MCP server as a subprocess and calls all nine tools through
the official SDK client, printing each result:

    PYTHONPATH=$PWD:$PWD/.venv-mcp-scratch/lib/python3.12/site-packages \\
        .venv-phase2/bin/python tests/manual_mcp.py \\
        --output-dir /tmp/meettrace-demo/recordings

(or any venv with both the app deps and `mcp` installed — see
docs/MCP.md). Requires the `mcp` package; without it this exits 2
with install guidance. Ollama is expected NOT to be running, so
ask_meetings should return the search-fallback answer.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


async def _run(output_dir: str) -> int:
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    env = dict(os.environ)
    path_entries = [str(REPO_ROOT)]
    if env.get("PYTHONPATH"):
        path_entries.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(path_entries)
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "app.mcp", "--output-dir", output_dir],
        env=env)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            listed = await session.list_tools()
            print(f"tools ({len(listed.tools)}): "
                  f"{[t.name for t in listed.tools]}")
            for tool in listed.tools:
                ro = (tool.annotations.read_only_hint
                      if tool.annotations else None)
                print(f"  {tool.name}: read_only_hint={ro}")

            async def call(name, args):
                result = await session.call_tool(name, args)
                assert not result.is_error, (name, result)
                text = "".join(c.text for c in result.content
                               if c.type == "text")
                payload = json.loads(text)
                print(f"\n== {name} {args} ==")
                print(json.dumps(payload, indent=2)[:2000])
                return payload

            sessions = await call("list_sessions", {"limit": 5})
            first = sessions["results"][0]["session_id"]
            await call("list_sessions",
                       {"limit": 5, "source": "teams"})
            await call("search_meetings",
                       {"query": "budget", "mode": "text"})
            await call("search_meetings",
                       {"query": "money", "mode": "hybrid"})
            await call("search_meetings",
                       {"query": "timeline", "mode": "semantic"})
            await call("get_session", {"session_id": first})
            await call("get_transcript",
                       {"session_id": first, "format": "text"})
            await call("get_transcript",
                       {"session_id": first, "format": "segments"})
            await call("list_speakers", {})
            await call("what_did_speaker_say",
                       {"speaker": "John", "query": "budget"})
            await call("ask_meetings",
                       {"question": "What is the budget for the pilot?"})
    print("\nMANUAL MCP CHECK: all nine tools answered, no errors.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Manual MCP check against real demo sessions.")
    parser.add_argument("--output-dir", default="/tmp/meettrace-demo/"
                        "recordings")
    args = parser.parse_args(argv)
    try:
        import mcp  # noqa: F401
    except ImportError:
        print("manual_mcp: the `mcp` package is not installed — "
              "`pip install mcp` (or use the scratch venv, see "
              "docs/MCP.md).", file=sys.stderr)
        return 2
    return asyncio.run(_run(args.output_dir))


if __name__ == "__main__":
    raise SystemExit(main())
