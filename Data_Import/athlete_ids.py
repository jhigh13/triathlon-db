# Data_Import/athlete_ids.py
import requests
from config.config import HEADERS, RANKING_URL
from Data_Upload.update_race_results import get_elite_programs, fetch_race_results

def get_gender_ids(ranking_id):
    response = requests.get(RANKING_URL.format(ranking_id=ranking_id), headers=HEADERS)
    response.raise_for_status()
    rankings = response.json().get("data", {}).get("rankings", [])
    return [entry["athlete_id"] for entry in rankings]

def get_olympic_athlete_ids():
    Tokyo_olympics = "131299"  # Tokyo 2020 Olympics

    elite_programs = get_elite_programs(Tokyo_olympics) # Tokyo 2020 Olympics
    ids = []
    for program_id, program_name in elite_programs:  # Unpack the tuple properly
        results_data = fetch_race_results(Tokyo_olympics, program_id)
        # Extract athlete_ids from the results array
        for result in results_data:
            ids.append(result["athlete_id"])
    
    return list(set(ids))  # Remove duplicates within Olympic results

def get_top_athlete_ids():
    male_ids = get_gender_ids(13)
    female_ids = get_gender_ids(14)

    olympic_ids = get_olympic_athlete_ids()
    
    # Combine all IDs and remove duplicates
    all_ids = male_ids + female_ids + olympic_ids
    unique_ids = list(set(all_ids))  # Remove duplicates across all sources
    
    print(f"Found {len(male_ids)} male ranking IDs, {len(female_ids)} female ranking IDs, {len(olympic_ids)} Olympic IDs")
    print(f"Total unique athlete IDs: {len(unique_ids)}")
    
    return unique_ids