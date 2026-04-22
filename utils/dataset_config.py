# Dataset and fold configuration registry.
# Maps fold_name -> {data_dir, max_num_agents}
# Used by train_biflow.py to auto-set data_dir and max_num_agents when not
# explicitly specified via CLI arguments.

FOLD_CONFIG = {
    # EgoTraj-TBD dataset
    "tbd": {
        "data_dir": "./data/egotraj",
        "max_num_agents": 16,
    },
    # T2FPV-ETH dataset (five leave-one-out folds)
    "eth": {
        "data_dir": "./data/t2fpv",
        "max_num_agents": 32,
    },
    "hotel": {
        "data_dir": "./data/t2fpv",
        "max_num_agents": 32,
    },
    "univ": {
        "data_dir": "./data/t2fpv",
        "max_num_agents": 32,
    },
    "zara1": {
        "data_dir": "./data/t2fpv",
        "max_num_agents": 32,
    },
    "zara2": {
        "data_dir": "./data/t2fpv",
        "max_num_agents": 32,
    },
}
