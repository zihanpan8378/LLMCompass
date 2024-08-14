cd ../..

# python -m ae.energy_experiments.scripts.test_memory_intensive_dummy --mode=sim --device=A100
# python -m ae.energy_experiments.scripts.test_memory_intensive_dummy --mode=sim --device=RTX4090
# python -m ae.energy_experiments.scripts.test_memory_intensive_dummy --mode=run --device=RTX4090

python -m ae.energy_experiments.scripts.test_memory_intensive_dummy --mode=run --device=RTX6000Ada
# python -m ae.energy_experiments.scripts.test_memory_intensive_dummy --mode=sim --device=RTX6000Ada

# python -m ae.energy_experiments.scripts.test_memory_intensive_dummy --mode=run --device=L4
# python -m ae.energy_experiments.scripts.test_memory_intensive_dummy --mode=sim --device=L4
