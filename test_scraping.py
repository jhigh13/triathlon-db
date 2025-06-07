# test_scraping.py - Quick test to examine website structure

import requests
from bs4 import BeautifulSoup

def test_single_page():
    """Test scraping a single ranking page to examine structure."""
    url = "https://old.triathlon.org/rankings/itu_world_triathlon_series_2016/male"
    
    print(f"Testing URL: {url}")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    response = requests.get(url, headers=headers, timeout=10)
    print(f"Status code: {response.status_code}")
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Look for different table patterns
        print("\n=== Looking for ranking tables ===")
        
        tables = soup.find_all('table')
        print(f"Found {len(tables)} tables total")
        
        for i, table in enumerate(tables):
            table_class = table.get('class', [])
            table_id = table.get('id', '')
            rows = table.find_all('tr')
            print(f"Table {i+1}: class={table_class}, id='{table_id}', rows={len(rows)}")
            
            # Check if this looks like a ranking table
            if len(rows) > 5:  # Likely ranking table
                header_row = rows[0] if rows else None
                if header_row:
                    headers = [th.get_text().strip() for th in header_row.find_all(['th', 'td'])]
                    print(f"  Headers: {headers}")
                
                # Show first few data rows
                for j, row in enumerate(rows[1:5]):  # Skip header, show first 20 data rows
                    cells = [td.get_text().strip() for td in row.find_all(['td', 'th'])]
                    if cells:
                        print(f"  Row {j+1}: {cells}")
                print("")

if __name__ == "__main__":
    test_single_page()
