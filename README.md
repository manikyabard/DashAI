# DashAI

<p align="center">
  <img width="150px" src="app\public\visualAI.png">
</p>

[![Open Source? Yes!](https://img.shields.io/badge/Version-0.1-green)](https://img.shields.io/badge/Version-0.1-green)
[![Open Source? Yes!](https://badgen.net/badge/Open%20Source%20%3F/Yes%21/blue?icon=github)](https://github.com/Naereen/badges/)
[![Maintenance](https://img.shields.io/badge/OS-Linux%2C%20Mac-red)](https://img.shields.io/badge/OS-Linux%2C%20Mac-red)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)
[![made-with-python](https://img.shields.io/badge/made%20with-Python%2C%20JS-brightgreen)](https://img.shields.io/badge/made%20with-Python%2C%20JS-brightgreen)
[![Documentation Status](https://img.shields.io/badge/Python-v3.6-blue)](https://img.shields.io/badge/Python-v3.6-blue)
[![Documentation Status](https://readthedocs.org/projects/ansicolortags/badge/?version=latest)](http://ansicolortags.readthedocs.io/?badge=latest)

## Installation

To install, just run the following commands in order.

### 1. Clone our Repo
```shell
git clone https://github.com/manikyabard/DashAI.git
```
### 2. cd into our Repo
```shell
cd DashAI/
```
### 3. Install the Dependencies
```shell
chmod +x install.sh && ./install.sh
```
### 4. Add DashAI to your PATH Environment Variable
#### 4.1. Open your bashrc file
```shell
sudo nano ~/.bashrc
```
#### 4.2. Scroll down to the bottom and paste the following line
```shell
$dashai = [PATH TO DASHAI]/run.sh
```
:warning: Replace _[PATH TO DASHAI]_ with the _path_

## Quick Start
Complete the [installation](#installation) steps as shown above. To start using DashAI, open a terminal window and type in the following command.
```shell
dashai
```
This will start a server and run the app. See [Working](#working) for next steps.

## Working
### Step 1: Choosing the task.
You first get to choose the type of application you want to create a model for. DashAI uses this information in later stages, to suggest architectures that have achieved state-of-the-art results in that task. You can choose one of four tasks: collaborative filtering, tabular, text, and vision.

### Step 2: Selecting the dataset.
You then provide DashAI the dataset you intend to use, and let DashAI know how to utilize the dataset best. DashAI then asks how to split the dataset (into training and validation sets), how to label it, and what transforms to apply on the dataset.

### Step 3: Selecting the model.
You then have to choose what architecture they want your model to have. DashAI provides architectures that have achieved state-of-the-art results in the task defined, but you may use any model built using PyTorch layers.

### Step 4: (Optional) Auto ML
At this point, you may choose one of three options:
- to use DashAI default hyper-parameters;
- to input hyper-parameter values of your choosing; or
- to use DashAI's auto ML component, _Verum_, to select the best possible hyper-parameter values.
In _Verum_, you may choose which hyper-parameters you want tuned, the number of experiments to be run, and whether the resulting values should automatically be applied to the model.

### Step 5: (Optional) Training the model.
DashAI then provides a simple training interface, where, if you have not chosen to utilize _Verum_'s automatic applying feature, you may input the hyper-parameter values required for training. You can also pick between generic training and [1-cycle training](https://arxiv.org/pdf/1803.09820.pdf).

### Step 6: (Optional) Explainability
You can then choose to visualize the attributions in the explainability component of DashAI, _DashInsights_. You may choose from a multitude of attribution-calculation algorithms, depending on your task. The visualizations can provide insight into why a model is predicting what it is predicting.

### Step 7: (Optional) Saving the model.
Finally, if users are so inclined, they can save their models as .pth files. We provide instructions on how to use these files in [the Deployment section of our Wiki](https://github.com/manikyabard/DashAI/wiki/6.-Deployment).

## More Info
Check out our [Wiki](https://github.com/manikyabard/DashAI/wiki)!

## Copyright
Copyright &copy; 2020 onward  
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this project's files except in compliance with the License. A copy of the License is provided in the LICENSE file in this repository.
