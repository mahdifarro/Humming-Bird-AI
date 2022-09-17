# Humming-Bird-AI
## Table of contents
* [General info](#general-info)
* [Requirements](#requirements)
* [Setup](#setup)
* [Details](#details)
* [Configuration](#configuration)
* [Acknowledgements](#acknowledgements)

## General info
This project allows you to train a AI agent that can play hummingbird game made with Unity. It's based on a course named ML-Agents: Hummingbirds.
I've also added a second camera for better observation.

## Requirements
Project is created with:
* Python version: 3.6+ 64-bit
* Unity version: 2020.1+
* Unity ML-Agents version: 1.0.0+ (see https://github.com/Unity-Technologies/ml-agents/blob/release_12_docs/docs/Installation.md)
* Python mlagents version: 2.6.1+

## Setup
To run this project,
1. Install dependencies:

```
$ pip install mlagents
```

2. Clone this repository.
3. Open project in Unity
4. go to project directory:
```
$ cd {project directory}\Humming-Bird-AI\HummingBird\Assets\ML-Agents\config
```
5. run below command to connect mlagents to unity environment. after seeing unity logo in command prompt, hit play in unity to start training the agent.
```
$ mlagents-learn ./Hummingbird.yaml --run-id={model name}
```

6. Once the training is complete, brain data are created in the folder:
```
./results/{model name}
```
This file `{model name}.onnx` contains the actual trained neural network. Copy this file to the Unity project folder and assign it to the `Model` field of the `Behaviour Parameters` . You can now hit play.


**P.S** if you wish to stop the game and continue the learning process you can simply stop the game in unity and resume it later using this command:
```
$ mlagents-learn ./Hummingbird.yaml --run-id={model name} --resume
```

## Configuration
While the system is able to learn the behavior quite well already, I found it is better to increase the complexity of the neural network a bit. Create a trainer_config.yaml file in the current folder, with the following content:
```
behaviors:
    Hummingbird::
        trainer_type: ppo    
        summary_freq: 10000
        max_steps: 5.0e6
        network_settings:
            hidden_units: 256

```
This will double the number of neurons per layer and add one more layer, allowing the system to learn a more complex function.

## Details
My **hb_01 model** was able to get 60% of flowers most of the times after 4 hours of training. you can change `Hummingbird.yaml` file to imrpove its current score.
This is a image of its totall reward during the training:
![reward](https://user-images.githubusercontent.com/45734322/190838375-c015b6ff-d9e0-44c9-8e25-8a3189ad142f.png)

### Observers
There are a totall of 10 observations that I've added in the code:
1. agent's local rotation (4 observations)

2. a normilized vector pointing to the nearest flower (3 observations)

3. a dot product that indicates whether the beak tip is in front of the flower (1 observation)

4. a dot product that indicates whether the beak is pointing toward the flower (1 observation)

5. the relative distance from the beak tip to the flower (1 observation)

Also, there are several ray perceptron 3D to collect data about the bird front, its distance from ceiling, and its distance from the ground.

### Actions
There are a totall of 5 actions:
1. Index 0: move vector x (+1 = right, -1 = left)
2. Index 1: move vector y (+1 = up, -1 = down)
3. Index 2: move vector z (+1 = forward, -1 = backward)
4. Index 3: pitch angle (+1 = pitch up, -1 = pitch down)
5. Index 4: yaw angle (+1 = turn right, -1 = turn left)

## Acknowledgements
- This project was based on [ML-Agents: Hummingbirds](https://learn.unity.com/course/ml-agents-hummingbirds).
