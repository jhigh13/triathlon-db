# Data_Import/historical_rankings_scraper.py
# Add project root to path for package imports
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import requests
import time
import logging
from bs4 import BeautifulSoup
from datetime import datetime, date
from typing import List, Dict, Tuple, Optional
import pandas as pd
from sqlalchemy import text
from Data_Import.database import get_engine
from config.config import HEADERS

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Configure logging to show debug statements for troubleshooting
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# URL patterns for historical rankings
WTCS_URL_PATTERN = "https://old.triathlon.org/rankings/world_triathlon_championship_series_{year}/{gender}"
WTR_URL_PATTERN = "https://old.triathlon.org/rankings/world_rankings_{year}/{gender}"

# Rate limiting configuration
RATE_LIMIT_DELAY = 1.5  # seconds between requests

# Historical ranking category mappings
RANKING_CATEGORIES = {
    "world_triathlon_championship_series_male": 15,
    "world_triathlon_championship_series_female": 16,
    "world_rankings_male": 13,  # Using Points List category for World Rankings
    "world_rankings_female": 14
}

class HistoricalRankingsScraper:
    """
    Scraper for historical triathlon rankings from old.triathlon.org
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        # Athlete ID matching utility
        from Data_Import.athlete_matching import match_athlete_id
        self.match_athlete_id = match_athlete_id
        
    def discover_available_rankings(self) -> List[Dict]:
        """
        Discover available historical ranking pages.
        
        Returns:
            List of dictionaries containing ranking metadata
        """
        logger.info("Starting discovery of available historical rankings...")
        
        available_rankings = []
        

        # ITU WTCS rankings (2016-2019, no data for 2020) use a different URL pattern
        ITU_WTCS_URL_PATTERN = "https://old.triathlon.org/rankings/itu_world_triathlon_series_{year}/{gender}"
        for year in range(2016, 2020):
            for gender in ["male", "female"]:
                url = ITU_WTCS_URL_PATTERN.format(year=year, gender=gender)
                ranking_info = self._check_ranking_availability(
                    url,
                    "world_triathlon_championship_series",
                    year,
                    gender
                )
                if ranking_info:
                    available_rankings.append(ranking_info)
                time.sleep(RATE_LIMIT_DELAY)

        # WTCS rankings (2021-2024) use the standard pattern
        for year in range(2021, 2025):
            for gender in ["male", "female"]:
                url = WTCS_URL_PATTERN.format(year=year, gender=gender)
                ranking_info = self._check_ranking_availability(
                    url,
                    "world_triathlon_championship_series",
                    year,
                    gender
                )
                if ranking_info:
                    available_rankings.append(ranking_info)
                time.sleep(RATE_LIMIT_DELAY)

        # World Rankings (2022-2024) - currently broken, skip for now
        # for year in range(2022, 2025):
        #     for gender in ["male", "female"]:
        #         url = WTR_URL_PATTERN.format(year=year, gender=gender)
        #         ranking_info = self._check_ranking_availability(
        #             url,
        #             "world_rankings",
        #             year,
        #             gender
        #         )
        #         if ranking_info:
        #             available_rankings.append(ranking_info)
        #         time.sleep(RATE_LIMIT_DELAY)
        
        logger.info(f"Discovery complete. Found {len(available_rankings)} available ranking pages.")
        return available_rankings
    
    def _check_ranking_availability(self, url: str, series: str, year: int, gender: str) -> Optional[Dict]:
        """
        Check if a ranking page is available and extract basic metadata.
        
        Args:
            url: URL to check
            series: Series name (world_triathlon_championship_series or world_rankings)
            year: Year of rankings
            gender: Gender category
            
        Returns:
            Dictionary with ranking metadata if available, None otherwise
        """
        try:
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Locate the ranking table by matching ITU or WTCS header patterns
                ranking_table = None
                itu_pattern = {'rank', 'first name', 'last name', 'yob', 'country', 'points'}
                wtcs_pattern = {'rank', 'given name', 'family name', 'noc', 'events', 'totalpoints'}
                for tbl in soup.find_all('table'):
                    header_row = tbl.find('tr')
                    if not header_row:
                        continue
                    cols = [cell.get_text().strip().lower() for cell in header_row.find_all(['th', 'td'])]
                    #logger.debug(f"Checking table headers at {url}: {cols}")
                    # match full ITU schema or WTCS schema
                    if itu_pattern.issubset(set(cols)) or wtcs_pattern.issubset(set(cols)):
                        ranking_table = tbl
                        break
                if not ranking_table:
                    tables = soup.find_all('table')
                    #logger.debug(f"No suitable ranking table found among {len(tables)} tables at {url}")
                    # Log header of the largest table for debugging
                    if tables:
                        largest = max(tables, key=lambda t: len(t.find_all('tr')))
                        hdr = largest.find('tr')
                        cols = [cell.get_text().strip() for cell in hdr.find_all(['th','td'])]
                        logger.debug(f"Largest table headers at {url}: {cols}")
                
                if ranking_table:
                    # Determine schema by inspecting header row
                    header_row = ranking_table.find('tr')
                    header_cols = [cell.get_text().strip().lower() for cell in header_row.find_all(['th', 'td'])]
                    records = []
                    # ITU schema: ['rank','first name','last name','yob','country','points']
                    itu_headers = {'rank', 'first name', 'last name', 'country', 'points'}
                    if itu_headers.issubset(set(header_cols)):
                        for row in ranking_table.find_all('tr')[1:]:
                            cells = row.find_all('td')
                            if len(cells) >= 6 and cells[0].get_text().strip().isdigit():
                                try:
                                    rec = {
                                        'rank': int(cells[0].get_text().strip()),
                                        'given_name': cells[1].get_text().strip(),
                                        'family_name': cells[2].get_text().strip(),
                                        'yob': cells[3].get_text().strip(),
                                        'noc': cells[4].get_text().strip(),
                                        'total_points': float(cells[5].get_text().strip())
                                    }
                                    records.append(rec)
                                except ValueError:
                                    continue
                    else:
                        # Default WTCS schema: require at least 8 columns
                        for row in ranking_table.find_all('tr')[1:]:
                            cells = row.find_all('td')
                            if len(cells) >= 8 and cells[1].get_text().strip().isdigit():
                                try:
                                    rec = {
                                        'rank': int(cells[1].get_text().strip()),
                                        'given_name': cells[2].get_text().strip(),
                                        'family_name': cells[3].get_text().strip(),
                                        'yob': cells[4].get_text().strip(),
                                        'noc': cells[5].get_text().strip(),
                                        'events': int(cells[6].get_text().strip()),
                                        'total_points': float(cells[7].get_text().strip())
                                    }
                                    records.append(rec)
                                except ValueError:
                                    continue
                    athlete_count = len(records)
                    ranking_info = {
                        'url': url,
                        'series': series,
                        'year': year,
                        'gender': gender,
                        'athlete_count': athlete_count,
                        'category_key': f"{series}_{gender}",
                        'ranking_cat_id': RANKING_CATEGORIES.get(f"{series}_{gender}"),
                        'ranking_cat_name': self._build_category_name(series, year, gender),
                        'status': 'available',
                        'athletes': records
                    }
                    logger.info(f"✓ Found rankings: {series} {year} {gender} ({athlete_count} athletes)")
                    return ranking_info
                else:
                    logger.warning(f"✗ No ranking table found: {url}")
                    
            elif response.status_code == 404:
                logger.debug(f"✗ Page not found: {url}")
            else:
                logger.warning(f"✗ HTTP {response.status_code}: {url}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"✗ Request failed for {url}: {str(e)}")
        except Exception as e:
            logger.error(f"✗ Unexpected error for {url}: {str(e)}")
            
        return None
    
    def _build_category_name(self, series: str, year: int, gender: str) -> str:
        """Build a human-readable category name."""
        series_names = {
            "world_triathlon_championship_series": "World Triathlon Championship Series",
            "world_rankings": "World Rankings"
        }
        gender_names = {"male": "Male", "female": "Female"}
        
        return f"{series_names.get(series, series)} {year} {gender_names.get(gender, gender)}"
    
    def upsert_rankings(self, rankings: List[Dict]):
        """
        Upsert scraped rankings into the existing athlete_rankings table.
        """
        engine = get_engine()
        today = date.today()
        upsert_sql = text(
            """
            INSERT INTO athlete_rankings
              (athlete_id, athlete_name, ranking_cat_name, ranking_cat_id,
               rank_position, total_points, retrieved_at)
            VALUES (:athlete_id, :athlete_name, :ranking_cat_name, :ranking_cat_id,
                    :rank_position, :total_points, :retrieved_at)
            ON CONFLICT (athlete_name, ranking_cat_name, retrieved_at)
            DO UPDATE SET
              rank_position = EXCLUDED.rank_position,
              total_points = EXCLUDED.total_points
            """
        )
        inserted = 0
        for r in rankings:
            cat_name = r['ranking_cat_name']
            cat_id = r['ranking_cat_id']
            with engine.begin() as conn:
                for athlete in r['athletes']:
                    full_name = f"{athlete['given_name']} {athlete['family_name']}"
                    aid = self.match_athlete_id(full_name)
                    params = {
                        'athlete_id': aid,
                        'athlete_name': full_name,
                        'ranking_cat_name': cat_name,
                        'ranking_cat_id': cat_id,
                        'rank_position': athlete['rank'],
                        'total_points': athlete['total_points'],
                        'retrieved_at': today
                    }
                    conn.execute(upsert_sql, params)
                    inserted += 1
        logger.info(f"Upsert of athlete rankings complete. {inserted} records processed.")

    def stage_rankings(self, rankings: List[Dict]):
        """
        Stage raw scraped rankings into the staging_rankings table.
        """
        engine = get_engine()
        today = date.today()
        insert_sql = text(
            """
            INSERT INTO staging_rankings
              (athlete_id, athlete_name, ranking_cat_name, ranking_cat_id,
               rank_position, total_points, retrieved_at)
            VALUES (:athlete_id, :athlete_name, :ranking_cat_name, :ranking_cat_id,
                    :rank_position, :total_points, :retrieved_at)
            """
        )
        inserted = 0
        with engine.begin() as conn:
            # Clear existing staging data for today's run
            conn.execute(text("DELETE FROM staging_rankings WHERE retrieved_at = :today"), {"today": today})
            for r in rankings:
                for athlete in r.get('athletes', []):
                    full_name = f"{athlete['given_name']} {athlete['family_name']}"
                    params = {
                        'athlete_id': None,
                        'athlete_name': full_name,
                        'ranking_cat_name': r['ranking_cat_name'],
                        'ranking_cat_id': r['ranking_cat_id'],
                        'rank_position': athlete['rank'],
                        'total_points': athlete['total_points'],
                        'retrieved_at': today
                    }
                    conn.execute(insert_sql, params)
                    inserted += 1
        logger.info(f"Staged {inserted} ranking records into staging_rankings.")
    
    def resolve_athlete_ids(self):
        """
        Resolve missing athlete_id in staging_rankings by matching existing or calling external API,
        and upsert new athlete records into athlete table.
        """
        engine = get_engine()
        today = date.today()
        # Load resolving utilities
        from Data_Upload.search_id import get_athlete_id as api_get_athlete_id
        from Data_Import.athlete_data import get_athlete_info_df
        from config.config import ATHLETE_TABLE_NAME
        # Statements
        select_names = text(
            """
            SELECT DISTINCT athlete_name
            FROM staging_rankings
            WHERE athlete_id IS NULL AND retrieved_at = :today
            """
        )
        update_stage = text(
            """
            UPDATE staging_rankings
            SET athlete_id = :aid
            WHERE athlete_name = :name AND retrieved_at = :today
            """
        )
        upsert_athlete = text(f"""
            INSERT INTO {ATHLETE_TABLE_NAME}
              (athlete_id, full_name, gender, country, age,
               category_to, category_coach, category_athlete,
               category_medical, category_paratriathlete)
            VALUES (:athlete_id, :full_name, :gender, :country, :age,
                    :category_to, :category_coach, :category_athlete,
                    :category_medical, :category_paratriathlete)
            ON CONFLICT (athlete_id) DO UPDATE SET
              full_name = EXCLUDED.full_name,
              gender = EXCLUDED.gender,
              country = EXCLUDED.country,
              age = EXCLUDED.age,
              category_to = EXCLUDED.category_to,
              category_coach = EXCLUDED.category_coach,
              category_athlete = EXCLUDED.category_athlete,
              category_medical = EXCLUDED.category_medical,
              category_paratriathlete = EXCLUDED.category_paratriathlete
        """
        )
        with engine.begin() as conn:
            rows = conn.execute(select_names, {"today": today}).fetchall()
            for (name,) in rows:
                # Try existing athlete match
                aid = self.match_athlete_id(name)
                if not aid:
                    try:
                        aid = api_get_athlete_id(name)
                        info_df = get_athlete_info_df(aid)
                        if not info_df.empty:
                            rec = info_df.to_dict(orient='records')[0]
                            conn.execute(upsert_athlete, rec)
                    except Exception as e:
                        logger.error(f"Failed to fetch/create athlete record for '{name}': {e}")
                        continue
                # Update staging_rankings with resolved id
                conn.execute(update_stage, {"aid": aid, "name": name, "today": today})
                logger.debug(f"Resolved athlete '{name}' to ID {aid}")
        logger.info("Athlete ID resolution complete.")

def main():
    """Main function to initialize schema then stage scraped rankings."""
    # Ensure database and staging table exist
    from Data_Import.database import initialize_database
    initialize_database()
    scraper = HistoricalRankingsScraper()
    available_rankings = scraper.discover_available_rankings()
    # Stage scraped rankings into staging_rankings table
    scraper.stage_rankings(available_rankings)
    # Resolve missing athlete_id values in staging table and upsert athlete records
    scraper.resolve_athlete_ids()
    
    print(f"\n=== DISCOVERY SUMMARY ===")
    print(f"Total available rankings: {len(available_rankings)}")
    
    # Group by series
    series_counts = {}
    for ranking in available_rankings:
        series = ranking['series']
        series_counts[series] = series_counts.get(series, 0) + 1
    
    for series, count in series_counts.items():
        print(f"  {series}: {count} rankings")
    
    # Show sample data
    if available_rankings:
        print(f"\n=== SAMPLE RANKINGS ===")
        for ranking in available_rankings[:5]:
            print(f"  {ranking['ranking_cat_name']}: {ranking['athlete_count']} athletes")
        sample = next(r for r in available_rankings if r['year']==2024 and r['gender']=='male')
        print(sample['athletes'][:3])
        return

if __name__ == "__main__":
    main()
