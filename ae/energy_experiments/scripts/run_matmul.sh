cd ../..

# python -m ae.energy_experiments.scripts.test_matmul --mode=sim --device=A100
# python -m ae.energy_experiments.scripts.test_matmul --mode=sim --device=RTX4090
# python -m ae.energy_experiments.scripts.test_matmul --mode=run --device=RTX4090

python -m ae.energy_experiments.scripts.test_matmul --mode=run --device=RTX6000Ada
# python -m ae.energy_experiments.scripts.test_matmul --mode=sim --device=RTX6000Ada

# python -m ae.energy_experiments.scripts.test_matmul --mode=run --device=L4
# python -m ae.energy_experiments.scripts.test_matmul --mode=sim --device=L4
