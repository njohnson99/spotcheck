# SpotCheck

**SpotCheck** is an open-source framework to evaluate blindspot discovery methods ("BDMs").  

![Images from an example experimental configuration generated using SpotCheck](spotcheck-example.png)

* In summary, SpotCheck works by generating "experimental configurations": synthetic image datasets & models with *known* true blindspots.
* In this repository, we release the code we used in our experiments to generate several experimental configurations with different datasets and true blindspot definitions.
* To learn more about SpotCheck, you can read our paper [on arXiv](https://arxiv.org/abs/2207.04104)!
* You can also check out our code for PlaneSpot (a simple BDM) [here](https://github.com/HazyResearch/domino/blob/main/domino/_slice/planespot.py).
---

# Setup

1. Run the script `./setup.sh`, which installs all of the packages necessary to run SpotCheck to a conda environment named `spotcheck`.
2. See our [demo notebook](https://github.com/njohnson99/spotcheck/blob/main/demo.ipynb), which shows how to use the `data_utils.SyntheticEC` class to randomly sample the semantic features and blindspots for a single experimental configuration.
