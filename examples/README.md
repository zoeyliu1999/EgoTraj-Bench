# Examples

Quick-start examples for EgoTraj-Bench.

## test_model.py — Model Sanity Check

Build the BiFlow model from a config file and verify construction.

```bash
# EgoTraj-TBD config
python examples/test_model.py --cfg cfg/biflow_k20.yml

# T2FPV-ETH config
python examples/test_model.py --cfg cfg/biflow_t2fpv_k20.yml
```

## test_pipeline.py — End-to-End Pipeline Test

Load dataset, build model, and verify data-model compatibility.

```bash
# EgoTraj-TBD
python examples/test_pipeline.py \
    --data_dir ./data/egotraj --fold_name tbd

# T2FPV-ETH (eth fold)
python examples/test_pipeline.py \
    --data_dir ./data/t2fpv --fold_name eth \
    --cfg cfg/biflow_t2fpv_k20.yml --data_source original_bal
```
