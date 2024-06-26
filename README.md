# SurvivalClustering :boom:

SurvivalClustering provides benchmarking for various techniques used in survival analysis. 
The dataset used for this purpose is the TCGA data. 

## References 🔗

This work is based on the Vadesc methodology, originally presented by [i6092467](https://github.com/i6092467). 
We have taken the foundational principles of this technology and incorporated the TCGA dataset for enhanced performance and analysis. For further details about the original work, please refer to the following resources:

- [🔎 Vadesc GitHub Repository](https://github.com/i6092467/vadesc)
- [📄 Original Vadesc Paper](https://openreview.net/forum?id=RQ428ZptQfU)

We acknowledge and appreciate the efforts of the original authors in developing the Vadesc technology, which has served as a stepping stone for our project.

## Development Environment :hammer_and_wrench:

- tensorflow==2.4.0 
- tensorflow-gpu==2.4.0
- CUDA==11.0
- cuDNN==8.0

## Dependencies :books:
Make sure to match the following library versions:

- numpy==1.19.2
- pandas==1.1.4
- matplotlib==3.3.2
- lifelines==0.25.6
- scikit-learn==0.22.2.post1

This benchmark has been reproduced in a Windows environment. :computer:

## Additional Environment Setup :wrench:

In some cases, you may need to install the Microsoft Visual C++ Build Tools. For installation, please follow this [link](https://www.microsoft.com/en-US/download/details.aspx?id=48159).

1. Access the link above, select the language according to your user environment, and proceed with the download. :link: 
2. Install Microsoft Build Tools 2015 version and reboot after installation. :wrench:  

## Execution Steps :running:

## Data Download 💾
To download the Preprocessed TCGA dataset, Click the following link: :  [DownloadLink](https://www.dropbox.com/scl/fo/s1iqh2cfvx2k7doz4idxw/AET5vQEuxY2BpRRg1R2JfqY?rlkey=zupbvyl19ii50q3tvnjwvw69d&dl=0)

## How to run❓
1. To download the Preprocessed TCGA dataset referenced above, please click the following link
2. In the terminal, run `example/py/python main.py`. :arrow_forward:

If you wish to run in a Jupyter Notebook environment, follow the steps below:

1. To download the Preprocessed TCGA dataset referenced above, please click the following link.
2. Then, run `example/notebook/clusting.ipynb`. :arrow_forward:
