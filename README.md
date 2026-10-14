# AI Water Conservation Assistant — MLOps Project

An end-to-end MLOps pipeline for detecting potentially unusual water consumption patterns and generating data-driven water conservation recommendations.

## 1. Project Overview

Water is a critical and limited resource, and inefficient consumption can occur without being immediately visible to users or facility managers.

This project develops an **AI Water Conservation Assistant** that analyzes historical water consumption data to identify consumption patterns that are potentially unusual compared with expected behavior.

The system is designed as an **end-to-end MLOps project**, rather than only a machine learning model.

The complete system covers:

- Data ingestion
- Data profiling
- Data cleaning
- Data validation
- Feature engineering
- Data versioning
- Data storage
- Time-based dataset splitting
- Anomaly detection
- Model experimentation
- Hyperparameter optimization
- Experiment tracking
- Model evaluation
- Recommendation generation
- Model deployment
- Monitoring
- Continuous improvement

The architecture is designed to be modular and extendable so that new datasets, features, models, validation rules, recommendation strategies, and deployment components can be introduced without redesigning the entire project.

---

# 2. Problem Statement

Traditional water monitoring systems often report consumption values but do not provide sufficient intelligence to determine whether a consumption pattern is unusual.

For example, a high consumption value may be caused by:

- Normal usage during a particular period
- A particular fixture being used
- A change in occupancy or activity
- A repeated consumption pattern
- An unusual consumption event
- A possible leakage or wastage situation

Therefore, simply identifying high consumption is not sufficient.

The project aims to learn consumption patterns from historical water usage and identify observations that differ significantly from expected behavior.

The system can then use the detected unusual patterns and their context to provide appropriate conservation-oriented recommendations.

---

# 3. Important Interpretation of the Model Output

This project is primarily an **anomaly detection system**.

The dataset does not provide a ground-truth label stating:

> "This observation represents confirmed water wastage."

Therefore, the system should not claim that every detected anomaly is definitely water wastage or leakage.

Instead, model outputs should be interpreted as:

> **Potentially unusual water consumption**

A detected anomaly may require additional investigation before concluding that water was actually wasted.

This distinction is important for both the machine learning methodology and the final recommendations.

---

# 4. Project Objectives

The major objectives are:

### Objective 1 — Data Understanding

Understand the structure, quality, distribution, and characteristics of the water consumption dataset.

### Objective 2 — Data Preparation

Clean and validate the dataset while preserving legitimate unusual observations.

### Objective 3 — Feature Engineering

Create meaningful temporal, consumption, and contextual features for anomaly detection.

### Objective 4 — Anomaly Detection

Develop machine learning models capable of identifying potentially unusual water consumption patterns.

### Objective 5 — Model Comparison

Compare baseline and advanced anomaly detection approaches.

### Objective 6 — Hyperparameter Optimization

Improve model performance through systematic hyperparameter optimization.

### Objective 7 — Experiment Tracking

Track datasets, parameters, metrics, models, and experiments.

### Objective 8 — Reproducibility

Ensure that data preparation, model development, and experiments can be reproduced.

### Objective 9 — Recommendations

Convert detected unusual consumption patterns into understandable water conservation recommendations.

### Objective 10 — Deployment and Monitoring

Deploy the final model and establish mechanisms for monitoring model and data behavior.

---

# 5. Dataset

The project uses the:

**HSB Living Lab Water Consumption Anonymized Dataset**

The dataset contains water consumption observations together with contextual and temporal information.

The current dataset contains the following attributes:

| Attribute | Description / Role |
|---|---|
| `timestamp` | Timestamp of the observation |
| `date` | Date component |
| `hour` | Hour of the observation |
| `day_of_week` | Numeric day-of-week information |
| `day_name` | Name of the day |
| `month` | Month |
| `year` | Year |
| `sensor_id` | Sensor identifier |
| `apartment` | Apartment/context identifier |
| `cluster_name` | Cluster/context information |
| `room_number` | Room identifier |
| `room_type` | Type of room |
| `type` | Water type |
| `attached_to` | Fixture/attachment associated with consumption |
| `value` | Water consumption value |
| `aggregated_value` | Aggregated/cumulative consumption value |

The dataset is retained in its original form under:

```text
data/raw/

The raw dataset should not be modified directly.
6. Dataset Characteristics
The dataset contains approximately 652,000 observations and 16 attributes.
The data covers multiple years of water consumption observations.
Initial data analysis identified:
- Missing contextual values
- Exact duplicate rows
- Repeated sensor/timestamp observations
- Different values occurring for some repeated sensor/timestamp combinations
- Highly right-skewed consumption values
- Potential cumulative-meter decreases
- Multiple temporal and contextual attributes
These observations are handled during the data preparation stage.
7. Data Preparation Strategy
The data preparation stage follows an important principle:
Clean invalid and redundant data without removing legitimate unusual consumption behavior.

This is especially important because unusual observations are the target of the anomaly detection system.
Therefore, a high consumption value should not automatically be considered bad data.
8. Data Cleaning Decisions
The current cleaning strategy includes the following decisions.
Data Issue	Treatment
Exact duplicate rows	Remove
Repeated sensor + timestamp with identical values	Investigate and remove only when confirmed redundant
Repeated sensor + timestamp with different values	Preserve and flag for investigation
Missing apartment	Preserve and represent as unknown/shared context
Missing cluster_name	Preserve and represent as unknown
Negative consumption	Validate and remove if confirmed invalid
Very high consumption	Preserve
Decreasing cumulative value	Flag and investigate
Invalid timestamp	Validate and correct/remove if necessary


The cleaning process must be reproducible and documented.
9. Data Validation
Data validation ensures that the cleaned dataset satisfies expected structural and quality constraints.
Validation includes checks such as:
Schema validation
- Required columns exist
- Expected data types are present
- Column names are correct
Missing-value validation
- Required fields are not unexpectedly missing
- Missing contextual fields are handled consistently
Value validation
- Consumption values are valid
- Negative values are detected
- Unexpected ranges are identified
Timestamp validation
- Timestamp values can be parsed
- Derived temporal fields are consistent
- Chronological ordering is maintained where required
Duplicate validation
- Exact duplicates are detected
- Sensor/timestamp conflicts are identified
Cumulative-value validation
- Unexpected decreases are detected
Pandera will be used for structured data validation.
10. Data Profiling
Data profiling is performed before and after cleaning.
Raw Data Profiling
The raw dataset is profiled to understand:
- Number of rows
- Number of columns
- Data types
- Missing values
- Duplicate values
- Unique values
- Statistical distributions
- Numerical ranges
- Categorical distributions
- Correlations
- Potential data-quality problems
Clean Data Profiling
After preprocessing, the dataset is profiled again.
The purpose is to verify:
- Whether duplicate records were removed correctly
- Whether missing-value treatment worked
- Whether invalid values were handled
- Whether distributions were unintentionally changed
- Whether important unusual observations were preserved
Profiling reports are stored under:
reports/profiling/

11. Feature Engineering
The raw dataset contains information that is useful for detecting consumption behavior, but not every raw column should automatically become a model feature.
Feature engineering converts the cleaned data into model-ready information.
11.1 Temporal Features
Potential temporal features include:
- Hour
- Day of week
- Month
- Weekend indicator
- Time-of-day categories
- Seasonal patterns
These features allow the model to understand that consumption expectations may differ between:
- Morning and night
- Weekdays and weekends
- Different months
- Different usage periods
11.2 Consumption Features
Consumption-related features may include:
- Current consumption
- Consumption change
- Historical consumption statistics
- Rolling averages
- Rolling standard deviation
- Recent consumption behavior
- Deviation from historical baseline
These features help the model distinguish normal behavior from unusual behavior.
11.3 Contextual Features
Contextual information may include:
- Room type
- Fixture / attachment
- Water type
- Sensor context
For example, normal water usage for a shower may be different from normal water usage for a toilet or kitchen tap.
12. Handling Identifier Columns
Identifiers require special consideration.
Columns such as:
- sensor_id
- apartment
- room_number
can be useful for grouping, historical baselines, reporting, and contextual analysis.
However, they should not automatically be used as direct model features.
Using raw identifiers carelessly can cause the model to memorize individual entities instead of learning general consumption behavior.
Therefore, identifiers will be evaluated based on their purpose and the model architecture.
13. Aggregated Consumption
aggregated_value represents an aggregated/cumulative measurement.
A raw cumulative value is not always the best model feature.
Instead, the pipeline may derive changes between consecutive observations to obtain consumption-related information such as:
meter_delta = current aggregated value
              -
              previous aggregated value

Unexpected decreases can be flagged for investigation.
14. Data Splitting Strategy
Because this project uses historical time-series data, a random train/test split is not appropriate.
Random splitting can cause future observations to appear in the training set while earlier observations appear in the test set.
That can introduce temporal leakage.
Therefore, the project uses a time-based split.
The planned split is:
Training:
Before 2022-01-01

Validation:
2022-01-01
through
before 2022-10-01

Test:
2022-10-01 onward

This allows the model to be evaluated on future observations that were not available during training.
15. Data Leakage Prevention
Data leakage is treated as a major concern.
Any transformation that learns information from the data must be fitted using training data only when appropriate.
Examples include:
- Scaling
- Encoding
- Statistical baselines
- Rolling statistics
- Threshold estimation
- Feature transformations
The validation and test periods must not influence training-time statistics.
This ensures that reported model performance is more representative of real deployment conditions.
16. Machine Learning Approach
The project focuses on anomaly detection.
There are no reliable supervised labels identifying confirmed water wastage for every observation.
Therefore, the initial approach is unsupervised/semi-supervised anomaly detection.
The planned models include:
1. Isolation Forest
2. Autoencoder
17. Model 1 — Isolation Forest
Isolation Forest will be used as the initial baseline anomaly detection model.
The basic idea is that unusual observations are easier to isolate than normal observations.
The model creates randomized partitions of the feature space.
Observations that require fewer partitions to isolate are considered more unusual.
Isolation Forest provides a strong baseline because it:
- Works with numerical features
- Does not require manually labelled anomaly data
- Can handle large datasets
- Provides anomaly scores
- Is relatively computationally efficient
The baseline model establishes a reference against which more advanced models can be compared.
18. Model 2 — Autoencoder
An autoencoder will be evaluated as a more advanced anomaly detection approach.
The model learns to reconstruct normal patterns.
The basic process is:
Input Features
      |
      v
   Encoder
      |
      v
 Latent Representation
      |
      v
   Decoder
      |
      v
Reconstructed Features

The reconstruction error can be used as an anomaly signal.
If an observation is significantly different from patterns learned during training, its reconstruction error may be higher.
The autoencoder therefore provides a different approach from Isolation Forest.
19. Model Comparison
The project will compare the anomaly detection approaches using appropriate evaluation strategies.
Potential evaluation considerations include:
- Anomaly score behavior
- Validation-set behavior
- Stability across time
- Precision where labelled/validated examples are available
- Recall where labelled/validated examples are available
- False-positive behavior
- Operational usefulness
- Computational requirements
Because the project lacks comprehensive ground-truth anomaly labels, model selection should not rely on a single conventional classification metric.
20. Anomaly Threshold
A model's anomaly score does not automatically determine what should be considered actionable.
A threshold must be defined.
The project may use statistically derived thresholds, such as a high percentile of scores or other validation-based thresholding methods.
Threshold selection must be performed without leaking information from the future test period.
The distinction is:
Model
  ↓
Anomaly Score
  ↓
Threshold
  ↓
Potentially Unusual
  ↓
Recommendation / Review

21. Recommendation System
The recommendation component is separate from the anomaly detection model.
The model determines whether consumption is potentially unusual.
The recommendation logic then uses the anomaly result together with contextual information.
Potential inputs to recommendation logic include:
- Anomaly score
- Consumption level
- Historical baseline
- Time of day
- Day of week
- Fixture / attachment
- Water type
- Room context
- Recent consumption behavior
- Persistence of unusual behavior
This separation is intentional.
The ML model should not directly generate arbitrary recommendations without a defined decision layer.
22. Recommendation Flow
The planned logic is:
Water Consumption Data
        |
        v
Data Cleaning
        |
        v
Feature Engineering
        |
        v
Trained Anomaly Detection Model
        |
        v
Anomaly Score
        |
        v
Threshold / Decision Logic
        |
        v
Potentially Unusual Consumption
        |
        v
Context Analysis
        |
        v
Recommendation Engine
        |
        v
Water Conservation Recommendation

For example, the system may distinguish between:
Unusual consumption
+
Shower context
+
Repeated occurrence
        |
        v
Review shower usage / possible prolonged usage

and:
Unusual consumption
+
Washing machine context
+
Expected time period
        |
        v
Potentially explainable usage

The exact recommendation rules will be finalized after the model and feature pipeline are implemented.
23. MLOps Architecture
The project is designed as an end-to-end MLOps system.
High-level architecture:
                    ┌──────────────────┐
                    │   Raw Dataset    │
                    └────────┬─────────┘
                             |
                             v
                    ┌──────────────────┐
                    │ Data Profiling   │
                    └────────┬─────────┘
                             |
                             v
                    ┌──────────────────┐
                    │ Data Cleaning    │
                    └────────┬─────────┘
                             |
                             v
                    ┌──────────────────┐
                    │ Data Validation  │
                    └────────┬─────────┘
                             |
                             v
                    ┌──────────────────┐
                    │ Feature Engineer │
                    └────────┬─────────┘
                             |
                             v
                    ┌──────────────────┐
                    │ Time-Based Split │
                    └────────┬─────────┘
                             |
                    ┌────────┴─────────┐
                    |                  |
                    v                  v
             Training Data      Validation/Test
                    |
                    v
             Model Training
                    |
          ┌─────────┴─────────┐
          |                   |
          v                   v
   Isolation Forest       Autoencoder
          |                   |
          └─────────┬─────────┘
                    |
                    v
             Model Evaluation
                    |
                    v
            Hyperparameter Tuning
                    |
                    v
             Experiment Tracking
                    |
                    v
              Model Selection
                    |
                    v
            Recommendation Layer
                    |
                    v
                Deployment
                    |
                    v
               Monitoring

24. MLOps Tools
The project uses different tools for different responsibilities.
Area	Tool
Programming	Python
Data processing	pandas
Numerical processing	NumPy
Data profiling	fg-data-profiling
Data validation	Pandera
Feature preprocessing	scikit-learn
Data storage	Parquet / project storage
Code versioning	Git
Code hosting	GitHub
Data versioning	DVC
Remote data storage	Google Drive / configured DVC remote
Baseline anomaly model	Isolation Forest
Deep anomaly model	Autoencoder
Hyperparameter optimization	Optuna
Experiment tracking	MLflow
Testing	pytest
API deployment	FastAPI
Server	Uvicorn
Containerization	Docker
CI/CD	GitHub Actions
Monitoring	To be implemented


Tools will be introduced progressively rather than all at once.
25. Data Versioning
Git is used for source-code versioning.
However, large datasets should not be treated like normal source-code files.
DVC is therefore used for data versioning.
The intended structure is:
GitHub
   |
   |-- Source code
   |-- Configuration
   |-- Documentation
   |-- DVC metadata
   |
   +--------------------+
                        |
                        v
                    DVC Remote
                        |
                        v
                  Dataset Storage

This allows the project to associate a specific code version with a specific dataset version.
26. Experiment Tracking
MLflow will be used to track machine learning experiments.
Experiments may include:
- Model type
- Hyperparameters
- Feature configuration
- Training period
- Validation period
- Anomaly threshold
- Metrics
- Model artifacts
Example:
Experiment
    |
    +-- Model: Isolation Forest
    +-- Features: Version 1
    +-- Parameters
    +-- Validation Results
    +-- Model Artifact

This makes it possible to compare experiments systematically instead of relying on manually recorded results.
27. Hyperparameter Optimization
Optuna will be used to optimize model hyperparameters.
The optimization process will be:
Define Search Space
       |
       v
Optuna Trial
       |
       v
Train Model
       |
       v
Evaluate
       |
       v
Return Objective Value
       |
       v
Optuna Selects Better Parameters
       |
       v
Repeat

The validation period will be used for model selection and hyperparameter tuning.
The final test period should remain untouched until final evaluation.
28. Model Registry and Lifecycle
The project will eventually maintain a model lifecycle such as:
Experiment
    ↓
Candidate Model
    ↓
Validation
    ↓
Selected Model
    ↓
Registered Model
    ↓
Deployment
    ↓
Monitoring
    ↓
Retraining

The purpose is to prevent an experimental model from automatically becoming the production model.
29. Deployment
The final model is intended to be exposed through an API.
A possible architecture is:
User / Application
       |
       v
    FastAPI
       |
       v
Preprocessing Pipeline
       |
       v
Trained Model
       |
       v
Anomaly Score
       |
       v
Recommendation Logic
       |
       v
API Response

The deployment layer should use the same preprocessing logic that was used during model development.
This prevents training-serving inconsistencies.
30. Monitoring
After deployment, the system should not be considered complete.
The deployed system should be monitored for:
Data drift
Changes in incoming consumption distributions.
Feature drift
Changes in the distributions of important model features.
Prediction behavior
Changes in anomaly rates or score distributions.
Model performance
Where validation information becomes available.
Pipeline failures
Problems in:
- Data ingestion
- Preprocessing
- Validation
- Model inference
- API operation
Monitoring allows the system to detect when retraining or investigation may be required.
31. Project Folder Structure
The project is designed to grow over multiple MLOps phases.
mlops_project/
│
├── .venv/                         # Local Python environment
│
├── data/
│   ├── raw/                       # Original immutable data
│   ├── interim/                   # Intermediate data
│   └── processed/                # Clean/model-ready data
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── profile.py
│   │   ├── clean.py
│   │   ├── validate.py
│   │   └── feature_engineering.py
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   └── build_features.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py
│   │   ├── predict.py
│   │   └── evaluate.py
│   │
│   └── utils/
│       ├── __init__.py
│       └── ...
│
├── configs/
│   ├── data.yaml
│   ├── features.yaml
│   └── model.yaml
│
├── reports/
│   ├── profiling/
│   ├── validation/
│   └── preprocessing/
│
├── models/
│   ├── preprocessing/
│   └── trained/
│
├── tests/
│   ├── data/
│   ├── features/
│   ├── models/
│   └── pipeline/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_model_experiments.ipynb
│
├── docs/
│   ├── data_dictionary.md
│   ├── preprocessing.md
│   ├── modeling.md
│   └── mlops_architecture.md
│
├── .dvc/
├── dvc.yaml
├── dvc.lock
├── params.yaml
│
├── requirements.txt
├── README.md
├── .gitignore
├── .dvcignore
│
├── Dockerfile
├── docker-compose.yml
│
└── .github/
    └── workflows/
        └── ci.yml

Some directories are introduced only when their corresponding phase begins.
32. Development Workflow
Development follows a controlled workflow.
Understand
    ↓
Implement one component
    ↓
Inspect the implementation
    ↓
Modify
    ↓
Run
    ↓
Debug
    ↓
Test
    ↓
Commit
    ↓
Move to next component

This prevents multiple untested components from being developed simultaneously.
33. Version Control Strategy
Git is used to track source-code and configuration changes.
A typical workflow is:
Make change
    ↓
Run tests
    ↓
Inspect results
    ↓
git status
    ↓
git add
    ↓
git commit
    ↓
git push

Commits should represent meaningful project changes.
Examples:
feat: add raw data profiling
feat: implement duplicate handling
feat: add Pandera validation
feat: add temporal features
feat: implement isolation forest baseline
feat: add MLflow tracking

34. Python Environment
The project uses a local virtual environment.
Recommended structure:
mlops_project/
│
├── .venv/
├── requirements.txt
└── ...

The .venv directory is not committed to Git.
The environment can be recreated using the pinned dependencies in:
requirements.txt

This supports reproducibility across development machines.
35. Installation
Clone the repository:
git clone <repository-url>
cd mlops_project

Create the virtual environment:
python -m venv .venv

Activate it on Windows:
.venv\Scripts\activate

Activate it on Linux/macOS:
source .venv/bin/activate

Upgrade pip:
python -m pip install --upgrade pip

Install dependencies:
pip install -r requirements.txt

36. Dependency Management
Package versions are pinned in:
requirements.txt

Exact versions are used to improve reproducibility.
New dependencies should not be added randomly.
When a new MLOps phase introduces a tool, its compatibility with the existing environment should be checked before the dependency is added.
The dependency environment will evolve as the project progresses through:
Data Preparation
        ↓
Data Versioning
        ↓
Model Development
        ↓
Experiment Tracking
        ↓
Deployment
        ↓
Monitoring

37. Testing Strategy
Testing is performed at multiple levels.
Unit Tests
Test individual functions such as:
- Cleaning functions
- Feature transformations
- Validation rules
- Utility functions
Data Tests
Verify:
- Schema
- Missing values
- Data ranges
- Duplicates
- Temporal consistency
Pipeline Tests
Verify that:
Raw Data
    ↓
Cleaning
    ↓
Validation
    ↓
Feature Engineering

works as expected.
Model Tests
Verify:
- Model can train
- Model can generate predictions
- Expected feature structure is maintained
- Model artifacts can be loaded
38. Reproducibility
Reproducibility is a core requirement of the project.
The project uses multiple layers of versioning:
Code
 ↓
Git

Data
 ↓
DVC

Dependencies
 ↓
requirements.txt

Configuration
 ↓
YAML / params

Experiments
 ↓
MLflow

Models
 ↓
Model artifacts / registry

Together these components allow experiments and pipeline results to be traced and reproduced.
39. Security and Privacy Considerations
The dataset is anonymized.
Nevertheless, the project should avoid unnecessarily exposing:
- Personal information
- Identifiable user information
- Credentials
- API keys
- Environment secrets
Credentials should never be committed to Git.
Environment-specific secrets should be stored outside source code.
40. Limitations
The project has several important limitations.
No direct ground-truth wastage labels
The model detects potentially unusual behavior rather than confirmed water wastage.
Unusual does not always mean waste
A high or unusual consumption event can have a legitimate explanation.
Historical data limitations
The quality of the model depends on the quality and representativeness of historical observations.
Identifier limitations
Sensor, apartment, and room identifiers may provide useful context but can also create memorization or generalization problems if used improperly.
Cumulative meter behavior
Unexpected decreases in cumulative values require investigation and may have multiple possible explanations.
Recommendation limitations
Recommendations are decision-support outputs and should not be treated as definitive proof of leakage or misuse.
41. Future Improvements
Potential future improvements include:
- Better anomaly labels
- Expert-validated anomaly datasets
- More sophisticated time-series models
- Online anomaly detection
- Real-time streaming data
- Improved recommendation ranking
- Personalized consumption baselines
- Explainable anomaly detection
- Automated retraining
- Advanced model monitoring
- Alert prioritization
- Dashboard development
- Production-scale deployment