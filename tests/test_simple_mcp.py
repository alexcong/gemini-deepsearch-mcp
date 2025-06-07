"""Simple test for MCP stdio server startup."""

import asyncio # Keep for asyncio.TimeoutError if used, though anyio.move_on_after may be better
import os
import subprocess # Not strictly needed if using anyio.open_process fully
import sys
import time # Not strictly needed for async tests

import pytest
import anyio # Import anyio


@pytest.mark.anyio
async def test_server_startup():
    """Test that the MCP server can start and stop cleanly."""
    print("Testing MCP server startup...")

    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    main_script_module = "gemini_deepsearch_mcp.main"
    uv_executable_path = os.path.expanduser("~/.local/bin/uv")

    cmd = [uv_executable_path, "run", "python", "-m", main_script_module]

    async with await anyio.open_process(
        cmd,
        stdout=subprocess.PIPE, # Can still use subprocess.PIPE with anyio
        stderr=subprocess.PIPE,
        cwd=parent_dir,
    ) as process:
        print("✓ Server process starting...")

        try:
            # Wait for the process to exit, with a timeout.
            # anyio.move_on_after is a context manager for timeouts.
            with anyio.move_on_after(3) as scope:
                await process.wait()

            if scope.cancelled_caught:
                # Process is still running after 3 seconds (timeout occurred)
                print("✓ Server is running after 3s.")
            else:
                # Process exited within 3 seconds
                stderr_output_bytes = await process.stderr.receive() # Use receive for anyio streams
                stderr_output = stderr_output_bytes.decode().strip()
                stdout_output_bytes = await process.stdout.receive()
                stdout_output = stdout_output_bytes.decode().strip()

                if process.returncode != 0:
                    print(f"✗ Server exited prematurely with code {process.returncode}.")
                    if stdout_output:
                        print(f"Stdout:\n{stdout_output}")
                    if stderr_output:
                        print(f"Stderr:\n{stderr_output}")
                    assert process.returncode == 0, \
                        f"Server exited prematurely with code {process.returncode}. Stderr: {stderr_output}"
                elif stderr_output:
                     print(f"✓ Server exited cleanly (code 0) but produced stderr output. This might indicate minor issues.")
                     print(f"Stderr:\n{stderr_output}")
                else:
                    print(f"✓ Server exited cleanly (code 0) without stderr output.")

        finally: # Ensure termination if it's still running
            if process.returncode is None:
                print("Terminating server...")
                process.terminate()
                with anyio.move_on_after(5) as term_scope:
                    await process.wait()

                if term_scope.cancelled_caught:
                    print("✗ Server termination timed out, killing.")
                    process.kill()
                    await process.wait() # Wait for kill to complete
                    print("✓ Server killed.")
                else:
                    print("✓ Server terminated cleanly.")


def test_imports():
    """Test that we can import the main module."""
    print("Testing imports...")

    try:
        # Add parent directory to path
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, parent_dir)

        # Test basic imports
        import main  # noqa: F401
        from gemini_deepsearch_mcp.main import deep_search, mcp

        print("✓ Can import main module")

        # Test that MCP server is created
        assert mcp is not None, "MCP server not found"
        print("✓ MCP server object exists")

        # Test that deep_search function exists
        assert deep_search is not None, "deep_search function not found"
        print("✓ deep_search function exists")

    except Exception as e:
        print(f"✗ Import test failed: {e}")
        assert False, f"Import test failed: {e}"


async def main():
    """Run all tests."""
    print("=== Simple MCP Server Tests ===\n")

    # Test imports first
    try:
        test_imports()
        import_success = True
    except Exception:
        import_success = False

    print()

    # Test server startup if imports work
    if import_success:
        startup_success = await test_server_startup()
    else:
        print("Skipping server startup test due to import failures")
        startup_success = False

    print()

    # Summary
    if import_success and startup_success:
        print("✅ All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
