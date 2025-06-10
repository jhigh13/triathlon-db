# Historical Triathlon Rankings Data Scraping - Functional Specification

## Goal
Build a web scraping system to obtain historical triathlon athlete rankings from the World Triathlon website and integrate this data into the existing PostgreSQL database to enhance machine learning model training data.

## Project Context
The triathlon database currently stores:
- **65,000+ race result rows** across **2,000+ athletes** and **3,400+ events**
- Current API-based rankings through the `ranking_import.py` module
- Individual race performance data with detailed split times and position tracking
- Existing `athlete_rankings` table with schema for rank position, points, and retrieval date

## Functional Requirements

### 1. Historical Rankings Data Discovery (Completed)
**Requirement**: Identify and catalog available historical ranking pages on the World Triathlon website.

**Implementation Details**:
- Target URL pattern for World Triathlon Series Rankings: `https://old.triathlon.org/rankings/world_triathlon_championship_series_{YEAR}/{GENDER}`
- Target URL pattern for World Triathlon Rankings Series: `https://old.triathlon.org/rankings/world_rankings_{YEAR}/{GENDER}`
- Gender values: `male`, `female`
- Year range: 2016-2024 for World Triathlon Series, 2022-2024 World Triathlon Rankings

**Acceptance Criteria**:
- [ ] System can programmatically discover available ranking years and genders
- [ ] Invalid/missing ranking pages are gracefully handled with appropriate logging
- [ ] Discovery process respects rate limiting (minimum 1-second delays between requests)

### 2. Web Scraping Infrastructure
**Requirement**: Build robust web scraping capability to extract ranking data from HTML pages.

**Implementation Details**:
- Use Python's `requests` + `BeautifulSoup` for HTML parsing
- Implement retry logic with exponential backoff for failed requests
- Add User-Agent headers to avoid bot detection
- Parse HTML tables containing ranking data

**Data Fields to Extract**:
- Rank position (integer)
- Athlete name (string, may be stored as first name + last name in two separate fields)
- Country (string)
- Points (float)
- Ranking year (derived from URL)

**Acceptance Criteria**:
- [ ] Successfully extracts all required data fields from ranking tables
- [ ] Handles missing or malformed data gracefully
- [ ] Implements proper error handling and logging for network failures
- [ ] Respects website's robots.txt and implements polite crawling practices

### 3. Data Validation and Quality Assurance
**Requirement**: Ensure scraped data meets quality standards before database insertion.

**Validation Rules**:
- Rank position must be positive integer
- Athlete name must not be empty
- Country code should be valid 3-letter ISO format (when available)
- Points must be non-negative float
- Year must be within reasonable range (2016-2024)

**Data Quality Checks**:
- Detect and flag duplicate athletes within same year/gender category
- Validate that rank positions are sequential (1, 2, 3, etc.)
- Check for reasonable point distributions (highest rank = highest points)

**Acceptance Criteria**:
- [ ] All scraped records pass validation before database insertion
- [ ] Invalid records are logged with specific error details
- [ ] Data quality report generated showing validation statistics
- [ ] Duplicates are detected and handled appropriately

### 4. Database Integration
**Requirement**: Store scraped historical rankings in the existing `athlete_rankings` table.

**Database Schema Mapping**:
```sql
athlete_rankings:
- athlete_id: Integer (requires lookup/creation)
- athlete_name: String (from scraped data)
- ranking_cat_name: String (e.g., "World Triathlon Championship Series 2024 Male")
- ranking_cat_id: Integer (use the table below for historical categories, ie. World Triathlon Championship Series would be 15 for WTCS Male)
- rank_position: Integer (from scraped data)
- total_points: Float (from scraped data)
- retrieved_at: Date (date of scraping)

Ranking ID	Category	Region	Program
11	Olympic	-	Elite Men
12	Olympic	-	Elite Women
13	Points List	-	Elite Men
14	Points List	-	Elite Women
15	World Triathlon Series	-	Elite Men
16	World Triathlon Series	-	Elite Women


```

**Data Processing Requirements**:
- Athlete name matching: Match scraped names to existing athletes in database
- Handle name variations (e.g., "Smith, John" vs "John Smith")
- Create new athlete records when no match found
- Generate unique ranking category IDs for historical data

**Acceptance Criteria**:
- [ ] All scraped data successfully inserted into `athlete_rankings` table
- [ ] Athlete name matching achieves >95% accuracy
- [ ] UPSERT operations handle conflicts gracefully
- [ ] New athlete records created with proper validation
- [ ] Historical rankings distinguished from current API rankings

### 5. Incremental Data Processing
**Requirement**: Support both full historical backfill and incremental updates.

**Processing Modes**:
- **Full Backfill**: Scrape all available historical years (one-time setup)
- **Annual Update**: Scrape latest completed year when new rankings published
- **Validation Mode**: Re-scrape and compare with existing data

**Progress Tracking**:
- Track scraping progress by year/gender combination
- Resume capability for interrupted scraping sessions
- Detailed logging of processed vs. skipped records

**Acceptance Criteria**:
- [ ] Full backfill completes successfully within reasonable timeframe (<2 hours)
- [ ] Incremental updates can be scheduled and automated
- [ ] Progress tracking prevents duplicate work
- [ ] Failed scraping sessions can be resumed without data loss

### 6. Integration with Existing ML Pipeline (Not a Current Requirement)
**Requirement**: Ensure scraped historical rankings data enhances machine learning model training.

**Feature Engineering Enhancements**:
- Historical ranking trends: Track athlete ranking changes over time
- Peak performance periods: Identify career-high ranking periods
- Consistency metrics: Calculate ranking volatility/stability
- Career trajectory: Model ranking progression patterns

**Data Relationships**:
- Link historical rankings to race results for validation
- Cross-reference ranking years with actual race performance
- Identify ranking-performance correlations

**Acceptance Criteria**:
- [ ] Historical rankings data accessible via existing database connections
- [ ] New features can be generated for ML model training
- [ ] Data format compatible with current ML pipeline (`model_pipeline.ipynb`)
- [ ] No performance degradation in existing data processing

### 7. Monitoring and Maintenance (Not a Current Requirement)
**Requirement**: Implement monitoring and alerting for ongoing data quality.

**Monitoring Capabilities**:
- Track scraping success/failure rates
- Monitor data quality metrics over time
- Alert on significant data anomalies
- Performance metrics (scraping speed, database insertion rates)

**Maintenance Tasks**:
- Annual review of available ranking years
- Website structure change detection
- Database optimization for historical data queries

**Acceptance Criteria**:
- [ ] Automated monitoring reports generated weekly
- [ ] Alerts triggered for scraping failures or data quality issues
- [ ] Performance metrics tracked and optimized
- [ ] Documentation updated with maintenance procedures

## Technical Implementation Notes

### Technology Stack
- **Web Scraping**: `requests`, `BeautifulSoup4`, `lxml`
- **Database**: Extend existing SQLAlchemy setup
- **Data Processing**: `pandas` for data manipulation
- **Logging**: Python `logging` module with structured output
- **Configuration**: Extend existing `config/config.py`

### File Structure
```
Data_Import/
├── historical_rankings_scraper.py  # Main scraping logic
├── athlete_matching.py             # Name matching algorithms
└── ranking_categories.py           # Historical category management

config/
└── scraping_config.py              # Scraping-specific configuration

docs/
└── historical-rankings-scraping.md # This specification
```

### Error Handling Strategy
- Network errors: Retry with exponential backoff
- Parsing errors: Log and continue with next record
- Database errors: Transaction rollback and detailed logging
- Data validation errors: Skip record and log validation failure

### Performance Considerations
- Implement rate limiting (1-2 seconds between requests)
- Use connection pooling for database operations
- Batch database insertions for efficiency
- Parallel processing for multiple years (with rate limiting)

## Data Quality and Validation

### Expected Data Volume
- Estimated 200-500 athletes per year/gender combination
- ~15 years of historical data = ~15,000-37,500 total records
- Manageable volume for quality validation and processing

### Data Completeness Targets
- **Athlete Name**: 100% completeness required
- **Rank Position**: 100% completeness required
- **Points**: 95% completeness target (some early years may lack points)
- **Country**: 90% completeness target (historical data may be incomplete)

### Accuracy Validation
- Cross-reference with known historical race results
- Spot-check rankings against archived triathlon websites
- Compare with existing API data where overlap exists

## Risk Mitigation

### Website Structure Changes
- **Risk**: Old triathlon website structure may change
- **Mitigation**: Implement flexible parsing with fallback methods
- **Monitoring**: Regular validation runs to detect structure changes

### Data Availability
- **Risk**: Historical ranking pages may be removed
- **Mitigation**: Priority scraping of oldest/most at-risk data
- **Backup**: Archive scraped HTML for future reference

### Performance Impact
- **Risk**: Large historical dataset may slow down existing queries
- **Mitigation**: Proper indexing on `retrieved_at` and `ranking_cat_id`
- **Optimization**: Consider partitioning by year if dataset grows large

## Success Metrics

### Data Coverage
- [ ] 90% of available historical years successfully scraped
- [ ] <5% data loss due to parsing errors
- [ ] 95% athlete name matching accuracy

### System Performance
- [ ] Scraping completes within 2 hours for full backfill
- [ ] Database queries maintain sub-second response times
- [ ] No degradation in existing ETL pipeline performance

### Data Quality
- [ ] 100% of records pass validation rules
- [ ] Ranking sequences are complete and logical
- [ ] Historical trends align with known triathlon history

This functional specification provides a comprehensive roadmap for implementing historical triathlon rankings data scraping while maintaining the high-quality standards of the existing triathlon database system.
