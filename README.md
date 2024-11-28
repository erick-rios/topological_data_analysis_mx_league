# TOPOLOGICAL DATA ANALYSIS LEAGUE MX
## Author: Erick Jesús Ríos González



This project applies topological data analysis (TDA) techniques to the statistics of Liga MX, the top professional football league in Mexico. The goal is to uncover hidden patterns and insights from player performance data, leveraging persistent homology and principal component analysis (PCA) to understand the underlying structure of the league's performance metrics.

<iframe src="reports/topological_data_analysis.html" width="100%" height="600px">
  This browser does not support inline frames.
</iframe>



## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         topological_data_analysis_mx_league and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── topological_data_analysis_mx_league   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes topological_data_analysis_mx_league a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```


## Setup

### Prerequisites

Before running the project, make sure you have the following tools installed:

- Python 3.9 or higher
- pip (Python package manager)

### Installing Dependencies

Install the required dependencies by running:

```bash
pip install -r requirements.txt
```
--------

