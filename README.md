# SurvivalClustering :boom:

SurvivalClustering provides benchmarking for various techniques used in survival analysis. 
The dataset used for this purpose is the TCGA data. 

## Development Environment :hammer_and_wrench:

- tensorflow 2.4.0 
- tensorflow-gpu 2.4.0

## Dependencies :books:
Make sure to match the following library versions:

- numpy=1.19.2
- pandas=1.1.4
- matplotlib=3.3.2
- lifelines=0.25.6
- scikit-learn=0.22.2.post1

This benchmark has been reproduced in a Windows environment. :computer:

In some cases, you may need to install the Microsoft Visual C++ Build Tools. For installation, please follow this [link](https://visualstudio.microsoft.com/ko/downloads/#build-tools-for-visual-studio-2019).

## Execution Steps :running:

1. For data preprocessing, execute `example/dataprocessing.ipynb`. :arrow_forward:
2. Then, in the terminal, run `python main.py`. :arrow_forward:

If you wish to run in a Jupyter Notebook environment, follow the steps below:

1. First, execute the dataprocessing as mentioned above.
2. Then, run `clusting.ipynb`. :arrow_forward:

Let's dive into the world of Survival Analysis! :swimmer:
