cd ../..

# python -m ae.energy_experiments.test_compute_indensive_dummy --mode=sim --device=A100
# python -m ae.energy_experiments.test_compute_indensive_dummy --mode=sim --device=RTX4090
# python -m ae.energy_experiments.test_compute_indensive_dummy --mode=run --device=RTX4090

python -m ae.energy_experiments.test_memory_indensive_dummy --mode=run --device=RTX6000Ada
# python -m ae.energy_experiments.test_compute_indensive_dummy --mode=sim --device=RTX6000Ada

# python -m ae.energy_experiments.test_compute_indensive_dummy --mode=run --device=L4
# python -m ae.energy_experiments.test_compute_indensive_dummy --mode=sim --device=L4
