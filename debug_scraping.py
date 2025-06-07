# Debug script for testing web scraping connection
import requests
from bs4 import BeautifulSoup
import time

def test_simple_connection():
    """Test basic connection to the website"""
    print("Testing basic connection...")
    
    # Test a known URL first
    test_url = "https://old.triathlon.org/rankings/world_triathlon_championship_series_2024/male"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
    }
    
    try:
        print(f"Requesting: {test_url}")
        response = requests.get(test_url, headers=headers, timeout=15)
        print(f"Status Code: {response.status_code}")
        print(f"Content Length: {len(response.content)} bytes")
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            print(f"Title: {soup.title.string if soup.title else 'No title'}")
            
            # Look for tables
            tables = soup.find_all('table')
            print(f"Found {len(tables)} tables")
            
            # Look for ranking-specific elements
            ranking_table = soup.find('table', class_='ranking-table')
            if ranking_table:
                print("Found table with class 'ranking-table'")
            else:
                print("No table with class 'ranking-table' found")
                
                # Try other common selectors
                alt_table = soup.find('table', {'id': 'ranking'})
                if alt_table:
                    print("Found table with id 'ranking'")
                else:
                    generic_table = soup.find('table', class_='table')
                    if generic_table:
                        print("Found table with class 'table'")
                    else:
                        print("No obvious ranking table found")
                        
                        # Show first few table elements for debugging
                        if tables:
                            print(f"First table classes: {tables[0].get('class', 'No class')}")
                            print(f"First table id: {tables[0].get('id', 'No id')}")
                            
                            # Show first few rows
                            rows = tables[0].find_all('tr')[:3]
                            for i, row in enumerate(rows):
                                cells = row.find_all(['td', 'th'])
                                cell_texts = [cell.get_text(strip=True) for cell in cells]
                                print(f"Row {i}: {cell_texts}")
            
            return True
        else:
            print(f"Failed with status {response.status_code}")
            print(f"Response headers: {dict(response.headers)}")
            return False
            
    except requests.exceptions.Timeout:
        print("Request timed out")
        return False
    except requests.exceptions.ConnectionError:
        print("Connection error")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

def test_multiple_urls():
    """Test multiple URLs to see which ones work"""
    test_urls = [
        "https://old.triathlon.org/rankings/world_triathlon_championship_series_2024/male",
        "https://old.triathlon.org/rankings/world_triathlon_championship_series_2023/male", 
        "https://old.triathlon.org/rankings/world_rankings_2024/male",
        "https://old.triathlon.org/rankings/world_triathlon_championship_series_2024/female",
    ]
    
    working_urls = []
    
    for url in test_urls:
        print(f"\n--- Testing: {url} ---")
        try:
            response = requests.get(url, timeout=10)
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                working_urls.append(url)
                print("✓ Working")
            else:
                print("✗ Not working")
        except Exception as e:
            print(f"✗ Error: {e}")
        
        time.sleep(1)  # Rate limiting
    
    print(f"\n=== SUMMARY ===")
    print(f"Working URLs: {len(working_urls)}")
    for url in working_urls:
        print(f"  ✓ {url}")

if __name__ == "__main__":
    print("=== DEBUGGING WEB SCRAPING ===")
    
    # Test basic connection first
    if test_simple_connection():
        print("\n" + "="*50)
        # If basic connection works, test multiple URLs
        test_multiple_urls()
    else:
        print("Basic connection failed. Check internet connection or website availability.")
