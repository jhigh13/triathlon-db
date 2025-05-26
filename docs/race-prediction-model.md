# Race Performance Prediction Model Specification

## Goal
Create a machine learning model that accurately predicts an athlete's finish time and position in their next race, outperforming the current baseline of using athlete mean times.

## Functional Requirements

### 1. Data Preprocessing
**Acceptance Criteria:**
- Filter data to only include races within last 5 years for relevancy
- Remove extreme outliers (>3σ) from finish times per race category
- Handle DNF/DNS appropriately as a separate classification task
- Ensure at least 3 prior races per athlete for reliable prediction

### 2. Feature Engineering Enhancement
**Acceptance Criteria:**
- Add race-specific features:
  - Course difficulty score based on median finish times
  - Weather data integration (temperature, humidity)
  - Season timing (peak vs off-season periods)
  - Field strength calculation per event
  - Split-specific rolling averages (swim/bike/run)
- Add athlete-specific features:
  - Training block indicators (days between races)
  - Performance trends (improving/declining)
  - Race experience in specific conditions
  - Discipline-specific performance trends
  - Recent DNF rates and patterns
  - Performance decay factor for older results
  - Seasonal performance adjustments

### 3. Model Architecture
**Acceptance Criteria:**
- Implement a multi-stage prediction pipeline:
  - Stage 1: DNF risk classifier (XGBoost)
  - Stage 2: Split-specific time predictions (swim/bike/run)
  - Stage 3: Overall finish time regression
  - Stage 4: Position prediction based on field strength
- Achieve MAE < 5% of mean finish time per category
- R² score > 0.8 for time prediction
- Position prediction within ±3 places for 80% of predictions
- Implement separate models for each race category
- Generate confidence intervals for all predictions

### 4. Validation Framework
**Acceptance Criteria:**
- Use time-based forward-chaining cross-validation
- Separate validation sets by race category
- Maintain athlete-specific grouping in splits
- Generate confidence intervals for predictions
- Monitor prediction drift over time
- Track feature importance across model versions

### 5. Database Integration
**Acceptance Criteria:**
- Store predictions in new `athlete_predictions` table with:
  - Split-specific predictions
  - Overall time and position predictions
  - Confidence intervals
  - Model version and timestamp
  - Feature importance metrics
  - Prediction accuracy tracking
- Automated model retraining triggers
- Performance monitoring dashboard integration

## Suggested Enhancements
1. Consider tracking athlete form/fatigue cycles
2. Add competitor field strength metrics
3. Implement seasonal performance adjustments
4. Track prediction accuracy over time
5. Add prediction confidence scores
6. Monitor feature importance stability
7. Implement automated model selection
8. Create separate models for different competition levels

## Risks and Mitigations
1. Data sparsity for new athletes
   - Use category averages as fallback
2. Course variation effects
   - Develop course difficulty normalization
3. Overfitting to specific athletes
   - Ensure robust cross-validation
4. Seasonal variations
   - Include temporal features and adjustments
