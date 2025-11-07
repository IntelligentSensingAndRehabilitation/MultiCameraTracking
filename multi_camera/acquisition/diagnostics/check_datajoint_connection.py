#!/usr/bin/env python3
"""
Check DataJoint database connectivity with timeout

This script tests the connection to the DataJoint database used by the
acquisition system. It uses a timeout wrapper to prevent hanging if the
database is unreachable or unresponsive.

Similar to check_active_cameras.py pattern.
"""
import datajoint as dj
import threading
import sys
from datetime import datetime


def check_connection(timeout_seconds=10):
    """
    Test DataJoint connection with timeout

    Args:
        timeout_seconds: Maximum time to wait for connection

    Returns:
        tuple: (connected: bool, error: str|None, timing: float|None)
    """
    result = {'connected': False, 'error': None, 'timing': None}

    def _test():
        try:
            start = datetime.now()
            dj.conn()  # Explicit connection test
            elapsed = (datetime.now() - start).total_seconds()
            result['connected'] = True
            result['timing'] = elapsed
        except Exception as e:
            result['error'] = str(e)

    thread = threading.Thread(target=_test, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)

    if thread.is_alive():
        return False, f"Connection timeout after {timeout_seconds}s", None

    return result['connected'], result.get('error'), result.get('timing')


if __name__ == "__main__":
    print("Checking DataJoint database connectivity...")
    print("")

    connected, error, timing = check_connection(timeout_seconds=10)

    if connected:
        print(f"✓ Connected to DataJoint database")
        print(f"  Connection time: {timing:.3f}s")
        print(f"  Database host: {dj.config.get('database.host', 'N/A')}")
        print(f"  Database name: {dj.config.get('database.database', 'N/A')}")
        sys.exit(0)
    else:
        print(f"✗ Failed to connect to DataJoint database")
        print(f"  Error: {error}")
        print("")
        print("Troubleshooting:")
        print("  1. Check if database server is running")
        print("  2. Verify network connectivity to database host")
        print("  3. Check credentials in .datajoint_config.json or env vars")
        print("  4. Ensure database service is accessible from Docker container")
        sys.exit(1)
