# Fair-MAMAB
Code for our [AAMAS '25](https://aamas2025.org/) paper, **Multi-agent Multi-armed Bandits with Minimum Reward Guarantee Fairness**.

### To reproduce our results,
we recommend creating a virtual environment and installing packages via `pip install -r requirements.txt`. 

The codes are described as follows.
- 'ef' folder corresponds to the proposed baseline of ExploreFirst algorithm. Please run the relevant script file inside this folder.
  - 'run_ef_2.sh' for 2 arms case on simulated data.
  - 'run_ef_k.sh' for $k>2$ arms case on simulated data.
  - 'run_mlens.sh' for the real-world experiment.
- 'ucb' folder corresponds to the proposed RewardFairUCB algorithm. Please run the relevant script file inside this folder.
  - 'run.sh' for the simulated data experiment.
  - 'run_mlens.sh' for the real-world experiment.
- 'dual' folder corresponds to the proposed Dual-Heuristics. Please run the relevant script file inside this folder.
  - 'run.sh' for the simulated data experiment.
  - 'run_mlens.sh' for the real-world experiment.

With the log files generated, the 'combined' folder has codes to generate plots comparing different algorithms and the explore-exploit trade-off of ExploreFirst.

*If you find this useful, consider giving a* ⭐ *to this repository & [citing our work](CITATION.cff).*
