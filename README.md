# Tabular Data Science Agent

A prototype Data Science Agent, that is able to solve Tabular Kaggle Data Science challenges, creating a trained model and a submission file for the test data, to be used for evaluation purposes. There are five different Data Science agent variants: ReACT, Hierarchical ReACT, Language Agent Tree Search (LATS), MCTS with outcome rewards, MCTS with hierarchical verifiable rewards, and Hierarchical MCTS + Verifiable rewards. 

## 🚀 Features

- **Hierarchical Agents**: Multi-stage ML pipeline automation with stage-specific tool restriction. Implemented for both ReACT and MCTS style approaches
- **MCTS Implementation**: Implementations of MCTS search variants for improved agent decision making, in LangChain
- **Function-to-Tool Conversion**: Decorators to automatically convert of arbitrary ML functions to LangChain tool format
- **Comprehensive ML Toolkit**: Reasonably comprehensive tools to perform Feature engineering, modeling, evaluation, hyperparameter tuning, data loading and other utility tasks

## 🛠 Installation

1. **Clone the repository:**
   ```bash
   git clone -b mcts https://github.com/Research-Adobe/tabular-datascience-agent.git
   ```

2. **Create a conda env and Install dependencies:**
   ```bash
   conda create --name ds_agent python=3.11
   conda activate ds_agent
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   Create a `.env` file with your API keys:
   ```env
    LANGSMITH_TRACING=false
    LANGSMITH_ENDPOINT=<>
    LANGSMITH_API_KEY=<>
    LANGSMITH_PROJECT=<>
    OPENAI_MODEL=gpt-4o
    OPENAI_BASE_URL=<>
    OPENAI_API_KEY=<>
    MCTS_OUTCOME_REWARD=False
    USE_FEEDBACK=True
    COMPETITION_NAME=customer_churn_prediction
   ```
    MCTS_OUTCOME_REWARD is a flag for if you want to use outcome based rewards with MCTS. Only relevant for MCTS

    USE_FEEDBACK is a flag for if you want to use textual feedback or not. Only relevant for MCTS and Hierarchical MCTS
    
    COMPETITION_NAME is the name of the competition that has been provided in the challenges.py folder


## 🚀 Quick Start

First add your competition information to the challenges.py file and set the COMPETITION_NAME to be the corresponding key, in the .env file


```python
CHALLENGES = {
    "your_competition": """
        Competition description and instructions...
        Data location: data/your_competition/
        Target: predict_column
        Submission format: id,prediction
    """
}
```

### 1. Basic React Agent

Run the basic React agent for automated ML pipeline:

```bash
python react_ds_agent.py
```

This agent:
- Creates a high-level plan for the ML task
- Uses all available tools without stage restrictions
- Provides interactive chat interface

### 2. Hierarchical React Agent

Run the hierarchical agent with stage-based progression. Also, provides an interactive chat interface.

```bash
python hierarchical_react_ds_agent.py
```

The stage_function_mapping json files map the stage name to the functions that are allowed to be used in that stage.

**Pipeline Stages:**
1. **Train Data Loading** - Load training dataset
2. **Test Data Loading** - Load test dataset  
3. **Combine Train/Test** - Merge datasets for consistent preprocessing
4. **Data Cleaning** - Handle missing values and data quality issues
5. **Feature Engineering** - Create, transform, and encode features
6. **Split Train/Test** - Separate combined dataset back into train/test
7. **Train Data to Features/Target** - Convert train data to ML format
8. **Test Data to Features** - Convert test data to ML format
9. **Modeling** - Train and tune ML models
10. **Create Submission** - Generate competition submission file

### 3. LATS (Language Agent Tree Search) Agent

Run LATS-powered agents for enhanced decision making:

```bash
python LATS/lats_ds_reflection.py
```

### 4. MCTS + Outcome Reward

First, set MCTS_OUTCOME_REWARD=True in the .env file

```bash
python LATS/lats_ds_verifiable_rewards.py
```

### 4. MCTS + Hierarchical Verfiable Reward

First, set MCTS_OUTCOME_REWARD=False in the .env file

```bash
python LATS/lats_ds_verifiable_rewards.py
```

### 5. Hierarchical MCTS + Verfiable Reward

This uses the same pipeline stages as the hierarchical ReACT agent.

```bash
python LATS/lats_hierarchical.py
```

## 🔧 Core Components

### Function-to-Tool Conversion

The framework automatically converts ML functions to LangChain tools using decorators:

```python
# Decorators available:
- make_get_and_set_tool: If the function needs to read from and write to scratchpad, use this decorator
- make_get_tool: If the function only needs to read from scratchpad, use this decorator
- make_set_tool: If the function only needs to write to scratchpad, use this decorator
- override_tool: If the function needs to ovveride the variable it read from the scratchpad, with its own result, use this decorator.
```

The function_to_decorator_map.json file maps the function name to the decorator that should be used to convert the function to a LangChain tool.

### Feature Engineering

Comprehensive feature engineering capabilities in `feature_engineering.py`:

- **Basic Operations**: Create numeric/categorical features, transformations
- **String Processing**: Pattern extraction, text manipulation
- **Aggregations**: Group-based statistics and aggregations
- **Encoding**: Categorical encoding, normalization
- **Advanced**: Interaction features, polynomial features

### Modeling

Support for multiple ML algorithms in `modeling.py`:

- **Tree-based**: XGBoost, LightGBM, CatBoost, Random Forest
- **Linear**: Logistic Regression, Linear Regression

### Hyperparameter Tuning

Automated hyperparameter optimization in `hyperparameter_tuning.py`:

- Grid search capabilities for hyperparameter tuning
- Model-specific parameter spaces
- Cross-validation integration


## 🔧 Configuration

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key
- `OPENAI_BASE_URL`: OpenAI API base URL  
- `OPENAI_MODEL`: Model to use (gpt-4, gpt-3.5-turbo, etc.)
- `COMPETITION_NAME`: Competition identifier from challenges.py

### Function Mappings

- `function_to_decorator_map.json`: Maps functions to appropriate decorators
- `stage_function_mapping.json`: Maps ML pipeline stages to available functions