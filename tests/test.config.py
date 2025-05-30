import os
import pytest
from config import TRI_API_KEY

def test_tri_api_key_env_var():
    # Ensure at least the env var exists (even if empty)
    assert 'TRI_API_KEY' in os.environ

def test_tri_api_key_not_empty():
    # Skip this if you haven't set the var locally
    if os.environ.get('TRI_API_KEY'):
        assert TRI_API_KEY != ''
    else:
        pytest.skip("TRI_API_KEY not set in environment")
