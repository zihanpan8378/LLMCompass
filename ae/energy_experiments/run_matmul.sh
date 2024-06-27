cd ../..

python -m ae.energy_experiments.test_matmul --mode=sim --device=A100
python -m ae.energy_experiments.test_matmul --mode=sim --device=RTX4090

#cd ae/figure5/ab
#python plot_matmul.py