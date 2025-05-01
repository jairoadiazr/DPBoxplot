# DPBoxplot
**Differentially Private Boxplots**

This repository contains the official implementation of **DPBoxplot**, te generate differentially private boxplots. 

📄 **Original Paper**: K. Ramsay, J. Diaz-Rodriguez. *"Differentially Private Boxplots"* [Available on arXiv](https://arxiv.org/abs/2405.20415) - To appear in ICML 2025 

*Despite the potential of differentially private data visualization to harmonize data analysis and privacy, research in this area remains underdeveloped. Boxplots are a widely popular visualization used for summarizing a dataset and for comparison of multiple datasets. Consequentially, we introduce a differentially private boxplot. We evaluate its effectiveness for displaying location, scale, skewness and tails of a given empirical distribution. In our theoretical exposition, we show that the location and scale of the boxplot are estimated with optimal sample complexity, and the skewness and tails are estimated consistently, which is not always the case for a boxplot naively constructed from a single existing differentially private quantile algorithm. As a byproduct of this exposition, we introduce several new results concerning private quantile estimation. In simulations, we show that this boxplot performs similarly to a non-private boxplot, and it outperforms the naive boxplot. Additionally, we conduct a real data analysis of Airbnb listings, which shows that comparable analysis can be achieved through differentially private boxplot visualization.*

## 📂 Repository Structure
- **`pboxplots.py`** – Core implementation containing the `pboxplots` function, a seaborn wrapper for differentially private boxplots.
- **`generated_figures/`** – Contains figures in the paper.
- **`private_quantiles/`** – Contains auxiliary functions to calculate differentially private quantiles.
- **`simulation_results/`** – Folder containing simulation results.
- **`data/`** – Folder containing the data for the case study.
- **Notebooks:**
  - `offline_experiments.ipynb` – Reproduces single distribution simulations from the paper.
  - `multiple_distributions_simulation.ipynb` – Reproduces multiple distribution simulations from the paper appendix.
  - `case_study.ipynb` – Generates boxplots in the case study from the paper.

## 📜 Citation
If you use this code or any of the data provided in this repository in your research, please cite the official paper:  
[arXiv:2405.20415](https://arxiv.org/abs/2405.20415) -- Soon to appear in ICML 2025
