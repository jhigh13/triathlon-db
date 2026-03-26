import requests
import time
import logging
import pandas as pd
from bs4 import BeautifulSoup
from datetime import date
from typing import List, Dict, Optional
from sqlalchemy import text
from dotenv import load_dotenv

try:
    from tri_analysis.database import get_engine, initialize_database
    from tri_analysis.athlete_matching import match_athlete_id
except ImportError:  # pragma: no cover
    from database import get_engine, initialize_database  # type: ignore
    from athlete_matching import match_athlete_id  # type: ignore

load_dotenv(override=True)

# Configure logging to show debug statements for troubleshooting
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress urllib3 and requests logging
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)

# URL patterns for historical rankings
WTCS_URL_PATTERN = "https://old.triathlon.org/rankings/world_triathlon_championship_series_{year}/{gender}"
WTR_URL_PATTERN = "https://old.triathlon.org/rankings/world_rankings_{year}/{gender}"

# Rate limiting configuration
RATE_LIMIT_DELAY = 1.0  # seconds between requests

# Historical ranking category mappings
RANKING_CATEGORIES = {
    "world_triathlon_championship_series_male": 15,
    "world_triathlon_championship_series_female": 16,
    "world_rankings_male": 13,
    "world_rankings_female": 14,
}

CATEGORY_LABELS = {
    "world_triathlon_championship_series_male": "World Triathlon Series - Male",
    "world_triathlon_championship_series_female": "World Triathlon Series - Female",
    "world_rankings_male": "World Rankings - Male",
    "world_rankings_female": "World Rankings - Female",
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
        self.match_athlete_id = match_athlete_id
        self._athlete_id_cache: Dict[str, Optional[int]] = {}

    def _lookup_athlete_id(self, full_name: str) -> Optional[int]:
        if full_name not in self._athlete_id_cache:
            self._athlete_id_cache[full_name] = self.match_athlete_id(full_name)
        return self._athlete_id_cache[full_name]
        
    def discover_available_rankings(self) -> List[Dict]:
        # Discover available historical ranking pages. Return list of dictionaries with metadata.

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

        for year in range(2022, 2025):
            for gender in ["male", "female"]:
                url = WTR_URL_PATTERN.format(year=year, gender=gender)
                ranking_info = self._check_ranking_availability(
                    url,
                    "world_rankings",
                    year,
                    gender
                )
                if ranking_info:
                    available_rankings.append(ranking_info)
                time.sleep(RATE_LIMIT_DELAY)
        
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
                    # match full ITU schema or WTCS schema
                    if itu_pattern.issubset(set(cols)) or wtcs_pattern.issubset(set(cols)):
                        ranking_table = tbl
                        break
                if not ranking_table:
                    tables = soup.find_all('table')
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
                    if athlete_count == 0:
                        logger.warning(f"✗ Parsed zero athletes: {url}")
                        return None
                    ranking_info = {
                        'url': url,
                        'series': series,
                        'year': year,
                        'gender': gender,
                        'athlete_count': athlete_count,
                        'category_key': f"{series}_{gender}",
                        'ranking_cat_id': RANKING_CATEGORIES.get(f"{series}_{gender}"),
                        'ranking_cat_name': self._build_category_name(series, gender),
                        'status': 'available',
                        'athletes': records
                    }
                    logger.info(f"✓ Found rankings: {series} {year} {gender} ({athlete_count} athletes)")
                    return ranking_info
                else:
                    logger.warning(f"✗ No ranking table found: {url}")
            else:
                logger.warning(f"✗ HTTP {response.status_code}: {url}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"✗ Request failed for {url}: {str(e)}")
        except Exception as e:
            logger.error(f"✗ Unexpected error for {url}: {str(e)}")
        return None
    
    def _build_category_name(self, series: str, gender: str) -> str:
        """Build a stable human-readable category name keyed by series+gender."""
        return CATEGORY_LABELS.get(f"{series}_{gender}", f"{series} - {gender}")

    @staticmethod
    def _snapshot_date_for_year(year: int) -> date:
        return date(year, 12, 31)
    
    def upsert_rankings(self, rankings: List[Dict]):
        """
        Upsert scraped rankings into the existing athlete_rankings table.
        """
        if not rankings:
            return

        engine = get_engine()
        upsert_sql = text(
            """
            INSERT INTO athlete_rankings
              (athlete_id, athlete_name, ranking_cat_name, ranking_cat_id,
               rank_position, total_points, year, retrieved_at)
            VALUES (:athlete_id, :athlete_name, :ranking_cat_name, :ranking_cat_id,
                    :rank_position, :total_points, :year, :retrieved_at)
            ON CONFLICT (athlete_name, ranking_cat_name, year, retrieved_at)
            DO UPDATE SET
              rank_position = EXCLUDED.rank_position,
              total_points = EXCLUDED.total_points,
              year = EXCLUDED.year
            """
        )
        inserted = 0
        for r in rankings:
            cat_name = r['ranking_cat_name']
            cat_id = r['ranking_cat_id']
            year = r['year']
            snapshot_date = self._snapshot_date_for_year(year)
            with engine.begin() as conn:
                for athlete in r['athletes']:
                    full_name = f"{athlete['given_name']} {athlete['family_name']}"
                    aid = athlete.get('athlete_id')
                    if aid is None:
                        aid = self._lookup_athlete_id(full_name)
                    params = {
                        'athlete_id': aid,
                        'athlete_name': full_name,
                        'ranking_cat_name': cat_name,
                        'ranking_cat_id': cat_id,
                        'rank_position': athlete['rank'],
                        'total_points': athlete['total_points'],
                        'year': year,
                        'retrieved_at': snapshot_date,
                    }
                    conn.execute(upsert_sql, params)
                    inserted += 1
        logger.info(f"Upsert of athlete rankings complete. {inserted} records processed.")

    def stage_rankings(self, rankings: List[Dict]):
        """
        Stage raw scraped rankings into the staging_rankings table.
        """
        engine = get_engine()
        insert_sql = text(
            """
            INSERT INTO staging_rankings
              (athlete_id, athlete_name, ranking_cat_name, ranking_cat_id,
               rank_position, total_points, year, retrieved_at)
            VALUES (:athlete_id, :athlete_name, :ranking_cat_name, :ranking_cat_id,
                    :rank_position, :total_points, :year, :retrieved_at)
            """
        )
        inserted = 0
        with engine.begin() as conn:
            if rankings:
                years = sorted({r["year"] for r in rankings})
                conn.execute(
                    text("DELETE FROM staging_rankings WHERE year = ANY(:years)"),
                    {"years": years},
                )
            for r in rankings:
                snapshot_date = self._snapshot_date_for_year(r['year'])
                for athlete in r.get('athletes', []):
                    full_name = f"{athlete['given_name']} {athlete['family_name']}"
                    params = {
                        'athlete_id': None,
                        'athlete_name': full_name,
                        'ranking_cat_name': r['ranking_cat_name'],
                        'ranking_cat_id': r['ranking_cat_id'],
                        'rank_position': athlete['rank'],
                        'total_points': athlete['total_points'],
                        'year': r['year'],
                        'retrieved_at': snapshot_date,
                    }
                    conn.execute(insert_sql, params)
                    inserted += 1
        logger.info(f"Staged {inserted} ranking records into staging_rankings.")

    def resolve_athlete_ids(self):
        """
        Resolve missing athlete_id in staging_rankings using the current athlete table.
        """
        engine = get_engine()
        select_names = text(
            """
            SELECT DISTINCT athlete_name, retrieved_at
            FROM staging_rankings
            WHERE athlete_id IS NULL
            """
        )
        update_stage = text(
            """
            UPDATE staging_rankings
            SET athlete_id = :aid
            WHERE athlete_name = :name AND retrieved_at = :retrieved_at
            """
        )

        with engine.begin() as conn:
            rows = conn.execute(select_names).fetchall()
            for name, retrieved_at in rows:
                aid = self._lookup_athlete_id(name)
                conn.execute(update_stage, {"aid": aid, "name": name, "retrieved_at": retrieved_at})

        logger.info("Athlete ID resolution complete.")

    def get_staged_rankings(self):
        """
        Retrieve processed rankings from the staging table to feed into upsert_rankings.
        Returns data in the format expected by upsert_rankings method.
        Note: The staging table doesn't have athlete details like given_name, family_name, country.
        The upsert_rankings method only needs athlete_name, so we'll extract name components.
        """
        engine = get_engine()
        
        # Query staging table grouped by category
        query = text("""
            SELECT DISTINCT 
                ranking_cat_name,
                ranking_cat_id,
                year
            FROM staging_rankings
            ORDER BY year, ranking_cat_name
        """)
        
        rankings = []
        
        with engine.connect() as conn:
            categories = conn.execute(query).fetchall()
            
            for cat in categories:
                # Get all athletes for this category
                athletes_query = text("""
                    SELECT 
                        athlete_id,
                        athlete_name,
                        rank_position as rank,
                        total_points
                    FROM staging_rankings
                    WHERE ranking_cat_name = :cat_name
                        AND ranking_cat_id = :cat_id
                        AND year = :year
                    ORDER BY rank_position
                """)
                
                athletes_result = conn.execute(athletes_query, {
                    'cat_name': cat.ranking_cat_name,
                    'cat_id': cat.ranking_cat_id,
                    'year': cat.year
                })
                
                athletes = []
                for athlete in athletes_result:
                    # Split athlete_name into given_name and family_name
                    name_parts = athlete.athlete_name.split(' ', 1)
                    given_name = name_parts[0] if len(name_parts) > 0 else ''
                    family_name = name_parts[1] if len(name_parts) > 1 else ''
                    
                    athletes.append({
                        'athlete_id': athlete.athlete_id,
                        'given_name': given_name,
                        'family_name': family_name,
                        'rank': athlete.rank,
                        'total_points': athlete.total_points
                    })
                
                if athletes:  # Only add if we have athletes
                    rankings.append({
                        'ranking_cat_name': cat.ranking_cat_name,
                        'ranking_cat_id': cat.ranking_cat_id,
                        'year': cat.year,
                        'athletes': athletes
                    })
        
        logger.info(f"Retrieved {len(rankings)} ranking categories from staging table")
        return rankings
    
    def run_full_pipeline(self, limit_rankings=None):
        """
        Run the complete historical rankings pipeline: discovery -> staging -> athlete resolution -> upsert.
        
        Args:
            limit_rankings (int, optional): Limit number of rankings processed for testing
        """
        logger.info("=== STARTING FULL ENHANCED PIPELINE ===")
        
        # Step 1: Discovery
        logger.info("Step 1: Discovering available rankings...")
        available_rankings = self.discover_available_rankings()
        
        if limit_rankings:
            available_rankings = available_rankings[:limit_rankings]
            logger.info(f"Limited to {len(available_rankings)} rankings for testing")
        
        logger.info(f"Found {len(available_rankings)} rankings to process")
        
        # Step 2: Staging
        logger.info("Step 2: Staging rankings data...")
        self.stage_rankings(available_rankings)
        
        # Step 3: Resolve athlete ids in staging rows
        logger.info("Step 3: Resolving athlete IDs...")
        self.resolve_athlete_ids()
          # Step 4: Retrieve staged rankings and upsert to athlete_rankings
        logger.info("Step 4: Retrieving staged rankings and upserting to athlete_rankings...")
        staged_rankings = self.get_staged_rankings()
        self.upsert_rankings(staged_rankings)
        
        logger.info("=== HISTORICAL PIPELINE COMPLETE ===")
        return available_rankings

def main():
    """Main function to run the enhanced historical rankings pipeline."""
    initialize_database()
    scraper = HistoricalRankingsScraper()
    
    print("=== ENHANCED HISTORICAL RANKINGS PIPELINE ===")
    print("This pipeline will:")
    print("1. Discover available historical rankings (2016-2024)")
    print("2. Stage raw ranking data")
    print("3. Resolve athlete IDs using the current athlete table")
    print("4. Upsert final rankings to athlete_rankings table")
    print()
    
    # Run the full enhanced pipeline
    available_rankings = scraper.run_full_pipeline()
    
    print(f"\n=== FINAL SUMMARY ===")
    print(f"Total rankings processed: {len(available_rankings)}")
    
    # Group by series and year for summary
    from collections import defaultdict
    by_series = defaultdict(list)
    total_athletes = 0
    
    for ranking in available_rankings:
        by_series[ranking['series']].append((ranking['year'], ranking['gender'], ranking['athlete_count']))
        total_athletes += ranking['athlete_count']
    
    for series, rankings in by_series.items():
        print(f"\n{series.replace('_', ' ').title()}:")
        for year, gender, count in sorted(rankings):
            print(f"  {year} {gender}: {count} athletes")
    
    print(f"\nTotal athletes in historical rankings: {total_athletes}")
    print("Enhanced pipeline complete!")

if __name__ == "__main__":
    main()
