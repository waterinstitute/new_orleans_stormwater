# Project Name

## Primary scientists and contributors
- Patrick Bodilly Kane (pkane@thewaterinstitute.org)  
- Nastaran Tebyanian (ntebyanian@thewaterinstitute.org)


## Project objective

Key Drivers of Vulnerability to Rainfall Flooding in New Orleans
Patrick Bodilly Kane, Nastaran Tebyanian, Daniel Gilles, Brett McMann1 & Jordan R. Fischbach

Keywords: Robust Decision Making, Decision Making Under Deep Uncertainty, Stormwater Management, SWMM, Spatial Decision Making.
Abstract
Future urban stormwater flood risk is determined by the confluence of both climate-driven changes in precipitation patterns and the effectiveness of flood mitigation systems, such as urban drainage and pump systems. This is especially true in coastal cities protected by levee systems like New Orleans, where even present-day rainfall would be enough to cause serious flooding in the absence of extensive stormwater drainage and pumping. However, while the uncertainties associated with climate change have been well studied, uncertainties in infrastructure performance and operation have received less attention. We investigated how these interrelated sets of uncertainties drive flood risk in New Orleans using a Robust Decision Making (RDM) approach. RDM is a framework for Decision Making Under Deep Uncertainty (DMDU) that leverages simulation models to facilitate exploration across many possible futures and the identification of decision-relevant scenarios. For our work, we leveraged a detailed Storm Water Management Model (SWMM) representation of the New Orleans urban stormwater management system to examine flood depths across the city when faced with different levels of future precipitation, sea-level rise, drainage pipe obstruction, and pumping system failure. We also estimated direct flood damage for each neighborhood in the city for this scenario ensemble. These damage estimates were then subjected to vulnerability analysis using scenario discovery—a technique designed to determine which combinations of uncertainties are most stressful to the system in terms of an outcome of interest (excess flood damage). Our results suggest that key drivers of vulnerability depend on geographic scale. Specifically, we find that possible climate-driven precipitation increases are the most important determinant of vulnerability at the citywide level. However, for some individual neighborhoods, infrastructure operation challenges under present day conditions are a more significant driver of vulnerability than possible climate-driven precipitation increases.


## Project scope
The repository contains code to replicate the vulnerability analysis for the paper “Key Drivers of Vulnerability to Rainfall Flooding in New Orleans”



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

