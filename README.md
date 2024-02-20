# Project Name

## Primary scientists and contributors
- Mark Bartlett (Mark.Bartlett@gmail.com) 

## Project objective
Machine Learning for...

## Project scope

### Problems solved

- Faster..
- Go beyond...
- Achieve high performance benchmark ...
- Reduce cost...

### Metrics for success

- MLE metrics:
    - F1 score
    - Inference time

- Data metrics:
    - Ability to benchmark against...
    - Train with observations...

- Reproducibility:
    - Ability to recreate data, ML, and deployment stages using pipeline

- Scientific soundness:
    - Data pre-processing - feature engineering based on state-of-the-science literature
    - Publish based on methodology developed for this projects

### Resources

- AWS...Google Cloud...etc.:
    - ML Workspace
        - Track experiments and register models according to the metrics in the `src/models` directory
        - Serve real-time endpoints
    - DevOps
        - Model training repository...
        - Pipelines repository
    - Computer Target (Databricks, etc.)
        - Jobs: run training and deployment jobs automatically
        - Feature store (to be used)
    - Storage
        - Critical piece to speed up inference as pre-processed data can be saved
    - Kubernetes..

### Data

- Source 1...
- Satellite...


## IT

IT has imposed policies....

## Project organization

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── config             <- Configuration files for the project.
    ├── build              <- Files for building environments
    │   ├── docker         <- Docker-compose, Dockerfile, requirements, etc. for the project.
    │   ├── k8s            <- Kubernetes files for the project
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │  
    │   │── deployment     <- Scripts to deploy the model as a service
    │   │   └── deploy_local.py
    │   │   └── deploy.py    
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── 

