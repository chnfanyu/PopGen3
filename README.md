# PopGen3

## Introduction

**PopGen** is a state-of-the-art open-source synthetic population generator for advanced travel demand modeling. Developed under the guidance of Professor Ram M. Pendyala at Arizona State University, PopGen utilizes a heuristic algorithm (IPU) to control and match both household-level and person-level attribute distributions.

## Key Features

- Advanced control of household and person variables across multiple geographic resolutions (region and geo).
- Command-line interface for simplified usage and enhanced computational efficiency.
- Fully compatible with Python 3, with updated dependencies.

## Run PopGen3 Online
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1j1Stb8IA8OfaoPRh232kId8hqi3dUtur?usp=sharing)
- [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/chnfanyu/PopGen3/HEAD)

## Contents

### Data Inputs

#### Survey sample data:
  `household_sample.csv`
  `person_sample.csv`

#### Population marginal data:

- **geo-level marginals:**  
     `person_marginals.csv`  
     `household_marginals.csv`  

- **region-level marginals:**  
  `region_person_marginals.csv`  
  `region_household_marginals.csv`  

#### Multi-Geographic Resolution Mapping Data:

  `region_geo_mapping.csv`  
  `geo_sample_mapping.csv`

Example of Multi-Geographic Resolution Levels:
- <small><i>Region Level:</i> Set as census county subdivision</small>
- <small><i>Geo Level:</i> Set as census tracts</small>
- <small><i>Sample Geo Level:</i> Set as Public Use Microdata Areas (PUMAs)</small>


### Configuration File 

The `configuration.yaml` file contains several key sections for the PopGen setup. 

#### Designed Input Keys

| **Project Wide Setting** | **Input Data Files** | **Scenario Settings** |
| --- | --- | --- |
| - synthesize<br>- name<br>- location | - entities<br>- column_names<br>- location | - description<br>- apply_region_controls<br>- control_variables<br>- parameters<br>- geos_to_synthesize<br>- outputs |

### Key Outputs

#### Sample Weighting Results
- `weights.csv`
#### Synthetic Population Generated
- `housing_synthetic.csv`
- `person_synthetic.csv`

