📁 Repository Overview: quadratic-refiner
The train_data folder contains intermediate results from the iterative MLP model.
Due to the final run being conducted with n=10, there are 9 intermediate files.
If you're looking for the version with 100 files as referenced in the paper, please refer to the commit history.

The test_experiment folder stores the final results of the iterative MLP.
Note: The configuration used here differs from the one in the paper, so absolute values may vary significantly.
Again, the version corresponding to the paper (with 100 files) can be found in the history.

The large_scale_baseline_experiment folder contains results from large-scale experiments using baseline MLP and residual MLP (Res-MLP) models under various hyperparameter settings.
These results are consistent with those reported in the paper.

The reinforcement learning (RL) component of this work relies on the framework provided by CleanRL and uses the WandB platform for experiment tracking. The task environment for this project is defined in the quadratic_env.py file within this repository. Additionally, some modifications were made to the ppo_continuous_action code to ensure compatibility with the task environment.

