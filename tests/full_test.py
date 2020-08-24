#!/usr/bin/env python3
import json
import torch
import fastai

from pathlib import Path

from text.learner import DashTextLearner
from tabular.learner import DashTabularLearner
from vision.learner import DashVisionLearner
from collabfilter.learner import DashCollabLearner
from core.train import DashTrain
from insights.DashInsights import DashInsights
from captum.insights.attr_vis import AttributionVisualizer

path = Path('./')
learner_class_map = {
	'collab': DashCollabLearner,
	'tabular': DashTabularLearner,
	'text': DashTextLearner,
	'vision': DashVisionLearner
}


def main():
	
	# fastai.torch_core.defaults.device = 'cpu'
	print('STEP 1: Creating the learner.')
	with open('./data/response.json') as f:
		response = json.load(f)
	application = response['task']
	save_dir = Path(response['save']['save_dir'])
	save_name = Path(response['save']['save_name'])
	learner_class = learner_class_map[application]
	learn = getattr(learner_class, f'create_{application}_learner')(response)
	print('Created learner; completed step 1.')
	learn.export('verum_test.pkl')
	data = learn.data
	print('STEP 2 (optional): Optimizing the hyper-parameters.')
	step_2 = False  # If step 2 done, then later use returned hyper-parameters.
	# Else, use default or mentioned hyper-parameters.
	try:
		import ax
		from verum.DashVerum import DashVerum
		step_2 = True
		with open('./data/verum.json') as f:
			response = json.load(f)
		verum = DashVerum(response, data, learn)
		learn, lr, num_epochs, moms = verum.veritize()
		print('Hyper-parameters optimized; completed step 2.')
	except ImportError:
		print('Skipping step 2 as the module `ax` is not installed.')

	print('STEP 3: Training the model.')
	if torch.cuda.is_available():
		with open('./data/train.json') as f:
			response = json.load(f)
		if step_2:
			response['fit']['epochs'] = num_epochs
			response['fit']['lr'] = lr
			response['fit_one_cycle']['max_lr'] = lr
			response['fit_one_cycle']['moms'] = str(moms)

		getattr(DashTrain, response['training']['type'])(response, learn)
		print('Trained model; completed step 3.')
	else:
		print('Skipping step 3 because there is no GPU.')

	print('STEP 4 (optional): Visualizing the attributions.')
	if(application == 'text' or application == 'vision'):
		insight = DashInsights(path, learn.data.batch_size, learn, application)
		fastai.torch_core.defaults.device = 'cpu'
		visualizer = AttributionVisualizer(
			models=[insight.model],
			score_func=insight.score_func,
			classes=insight.data.classes,
			features=insight.features,
			dataset=insight.formatted_data_iter(),
			application=insight.application
		)

		visualizer.serve(debug=True)
		print('Completed visualization; completed step 4.')
	else:
		print("Visualization is not possible for this application")

	print('STEP 5: Saving the model.')
	# save_path = save_dir / save_name
	# if not save_dir.exists():
	# 	save_dir.mkdir()
	# learn.export(save_path)
	print('Saved the model; completed step 5. Congratulations!')
	print('(Not actually saving right now; uncomment the relevant lines if needed.)')
	print('Load the model again with the following code:', end='\n\n')
	print(f'\tlearn = load_learner(path={save_dir!r}, file={save_name!r})', end='\n\n')
	print('-' * 50)
	print('Now we need to add production-serving.')


if __name__ == '__main__':
	main()
