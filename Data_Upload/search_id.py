# athlete_search.py
import requests
from config import HEADERS, ATHLETE_SEARCH_URL

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

# Example (for testing):
# athlete_id = get_athlete_id("Hayden Wilde")
# print(athlete_id)