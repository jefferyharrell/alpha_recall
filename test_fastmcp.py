#!/usr/bin/env python3
"""
Test the new FastMCP server for alpha-recall
"""

import subprocess
import sys
import time
import signal
import os

def test_fastmcp_server():
    """Test the FastMCP server startup and basic functionality"""
    print("🧪 Testing alpha-recall FastMCP server...")
    
    # Start the server in the background
    print("🚀 Starting FastMCP server...")
    
    env = os.environ.copy()
    env["MCP_TRANSPORT"] = "streamable-http"
    env["FASTMCP_HOST"] = "localhost" 
    env["FASTMCP_PORT"] = "6006"  # Use different port to avoid conflicts
    
    try:
        server_process = subprocess.Popen([
            sys.executable, "-m", "alpha_recall.fastmcp_server"
        ], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Give server time to start
        time.sleep(3)
        
        # Check if server is running
        if server_process.poll() is None:
            print("✅ FastMCP server started successfully")
            
            # Test basic connection (you could add actual MCP client test here)
            print("🔍 Server is running on localhost:6006")
            
            # Clean shutdown
            server_process.terminate()
            server_process.wait(timeout=5)
            print("🛑 Server shut down cleanly")
            return True
        else:
            stdout, stderr = server_process.communicate()
            print(f"❌ Server failed to start")
            print(f"STDOUT: {stdout.decode()}")
            print(f"STDERR: {stderr.decode()}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing server: {e}")
        if 'server_process' in locals():
            try:
                server_process.terminate()
            except:
                pass
        return False

if __name__ == "__main__":
    success = test_fastmcp_server()
    if success:
        print("🎉 FastMCP server test passed!")
    else:
        print("💥 FastMCP server test failed!")
        sys.exit(1)