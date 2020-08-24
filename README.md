# DashAI

## Installation

We're working on this!

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
