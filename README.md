

# How Safe Am I Given What I See?

**Calibrated Prediction of Safety Chances for Image-Controlled Autonomy**

This is the code for the paper: "How Safe Am I Given What I See?  Calibrated Prediction of Safety Chances for Image-Controlled Autonomy"

## Prerequisites

```python
pip install -r requirements.txt
```

The version of gym is 0.21.0 and box2d-py is 2.3.8.

## Train the controllers

The code to train the racing car controller is in the directory RacingCarController(https://github.com/andywu0913/OpenAI-GYM-CarRacing-DQN) 

The code to train the cart pole in the directory CartPoleController(https://github.com/fedebotu/vision-cartpole-dqn)

## Data generation

Before collecting data, we need to edit the car_racing.py files in PYTHON_PATH/site-packages/gym/envs/box2d/car_racing.py to get the position of the car at each moment.

In def step(self,cation), just change the return value to:

```python
return self.state, step_reward, done, [self.car.hull.position,self.road_poly,self.observation_space]
```
or directly substitute the file with EditedGym/car_racing.py

In the experiments on the racing car, we used 6 different models and collected 80K training samples, 20K test samples, 20K calibration samples, and 20K validation samples for each controller. 

The example is here:

```bash
python RacingCar/gene_data.py --n=80000 -d="RacingCar/data/train/controller_6/" -c="RacingCar/models/trial_600.h5" -s=0
```
where n is the number of samples, d is the dir of output path, c is the path of DQN models, and s is the random seed.

For the cart pole experiments, we used 3 different models and collect 30K training samples, 30K test samples, 30K calibration samples, and 30K validation samples for each controller

```python
    return np.array(self.state, dtype=np.float32), reward, done, theta
```

```bash
python CartPole/gene_data.py --n=30000 -d="CartPole/data/train/controller_1/" -c="CartPole/models/policy_net_best1.pt" -s=0
```

## Evaluator training

```bash
python RacingCar/evaluator/train.py --log="RacingCar/evaluator/logs/" --train="RacingCar/data/train/" --test="RacingCar/data/test/"
```


```bash
python CartPole/evaluator/train.py  --log="CartPole/evaluator/logs/" --train="CartPole/data/train/" --test="CartPole/data/train/"
```

## VAE training
```bash
python RacingCar/vae/train_unsafe_vae.py --log="RacingCar/vae/unsafe/" --train="RacingCar/data/train/" --test="RacingCar/data/test/" --eva="RacingCar/models/eva.tar"
```
