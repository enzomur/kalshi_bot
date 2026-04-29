"""Trading strategies that emit Signal objects.

IMPORTANT: Strategies MUST NOT import from execution/.
They emit Signals only - the Risk Engine is the sole gatekeeper to execution.
"""
