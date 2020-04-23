# Project

## Deep ML Project

### Training an hyper network in order to predict future datapoint in an interpretable manner.
### Instead of training the network to predict the next values we train a network to predict coefficients.

In order to run the project:
* Install all requirements from resources/requirements.txt (pip install -r requirements.txt)
* From the root directory run python ./src/main.py

Optional parameters :
* --num_experiments: How many experiments to run, default 10.
* --seq_length: Time series sequance length (how many data points to feed to the networks). default 168.
* --horizon: How many data point in the future to predict. default=24.
* --batch_size': Batch size. default=16
* --epochs': Number of epochs per experiment. type=int default=50,
* --dataset': Which dataset to load, either stocks or traffic. default='traffic',
