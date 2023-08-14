
# We use a simulator to test our method.

We use Blades, which is a simulator for Byzantine-robust federated Learning with Attacks and Defenses Experimental Simulation.

For the aggregation method, we change the code in /src/blades/aggregators/dnc.py to randomized SVD

For the encryption, we change the codes in /src/blades/clients/client.py, /src/blades/servers/server.py, /src/blades/core/simulator.py

# How to run the code?

`
cd scripts 

python main.py --config_path ../config/example.yaml
`

# Requirements
matplotlib>=3.4.1

numpy>=1.19.4

pre-commit

ray>=1.0.0

requests>=2.27.1

ruamel.yaml>=0.17.21

ruamel.yaml.clib>=0.2.6

scikit-learn>=1.0.2

scipy>=1.5.4

setuptools

sklearn>=0.0

torch>=1.10.2

torchvision>=0.11.3
tqdm

# Reference
Blades ： https://github.com/lishenghui/blades
