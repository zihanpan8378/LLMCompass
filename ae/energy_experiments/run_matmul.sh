cd ../..

# python -m ae.energy_experiments.test_matmul --mode=sim --device=A100
# python -m ae.energy_experiments.test_matmul --mode=sim --device=RTX4090
# python -m ae.energy_experiments.test_matmul --mode=run --device=RTX4090

python -m ae.energy_experiments.test_matmul --mode=run --device=RTX6000Ada
# python -m ae.energy_experiments.test_matmul --mode=sim --device=RTX6000Ada

#cd ae/figure5/ab
#python plot_matmul.py