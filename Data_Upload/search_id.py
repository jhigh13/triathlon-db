# athlete_search.py
from dotenv import load_dotenv
import sys
import os
# Add project root to path for package imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
load_dotenv()

import requests
from config.config import HEADERS, ATHLETE_SEARCH_URL

def get_athlete_id(athlete_name):
    """
    Fetch athlete ID based on a user-provided name.
    
    Parameters:
        athlete_name (str): The name of the athlete to search for.
    
    Returns:
        int: Athlete ID if found.
    
    Raises:
        ValueError: If the athlete is not found.
        Exception: If the API call fails.
    """
    params = {"query": athlete_name}
    response = requests.get(ATHLETE_SEARCH_URL, params=params, headers=HEADERS)
    
    if response.status_code == 200:
        data = response.json().get("data", [])
        if data:
            return data[0]["athlete_id"]
        else:
            raise ValueError(f"Athlete '{athlete_name}' not found.")
    else:
        raise Exception(f"Search API failed with status code {response.status_code}")

def get_full_search_results(athlete_name):
    """
    Get full search results for debugging purposes.
    """
    params = {"query": athlete_name}
    response = requests.get(ATHLETE_SEARCH_URL, params=params, headers=HEADERS)
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Search API failed with status code {response.status_code}")

# Example (for testing):
# athlete_id = get_athlete_id("Hayden Wilde")
# print(athlete_id)

if __name__ == "__main__":
    import sys
    import json
    if len(sys.argv) != 2:
        print("Usage: python search_id.py \"Athlete Name\"")
        sys.exit(1)
    
    athlete_name = sys.argv[1]
    try:
        # Show full search results
        full_results = get_full_search_results(athlete_name)
        print(f"Full API response for '{athlete_name}':")
        print(json.dumps(full_results, indent=2))
        
        # Try to get athlete ID
        athlete_id = get_athlete_id(athlete_name)
        print(f"\nAthlete ID for '{athlete_name}': {athlete_id}")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"API Error: {e}")