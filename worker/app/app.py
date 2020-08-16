from flask import Flask, render_template, request, jsonify
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
from flask_socketio import SocketIO, emit


app = Flask(__name__, template_folder="template")
socketio = SocketIO(app) # socket object to setup websocket connection
learner = None

path = Path('./')
learner_class_map = {
	'collab': DashCollabLearner,
	'tabular': DashTabularLearner,
	'text': DashTextLearner,
	'vision': DashVisionLearner
}


@app.route("/", methods=['GET'])
def helper():
    return render_template("helper.html")

@app.route("/generate", methods=['POST'])
def generate():
    res = {
        "status": "COMPLETE",
        "message": "",
        "payload": []
    }
    response = request.form.get()
    application = response['task']
	save_dir = Path(response['save']['save_dir'])
	save_name = Path(response['save']['save_name'])
	learner_class = learner_class_map[application]
	learn = getattr(learner_class, f'create_{application}_learner')(response)
	emit(print('-'*10, 'Created learner', '-'*10))

    emit('STEP 2 (optional): Optimizing the hyper-parameters.')
	step_2 = False  # If step 2 done, then later use returned hyper-parameters.
	# Else, use default or mentioned hyper-parameters.
	try:
		import ax
		from verum.DashVerum import DashVerum
		step_2 = True
		with open('../../data/verum.json') as f:
			response = json.load(f)
		verum = DashVerum(response, learn)
		learn, metric, lr, num_epochs, moms = verum.veritize()
		emit('Hyper-parameters optimized; completed step 2.')
	except ImportError:
		emit('Skipping step 2 as the module `ax` is not installed.')

    emit('STEP 3: Training the model.')
	if torch.cuda.is_available():
		with open('../../data/train.json') as f:
			response = json.load(f)
		if step_2:
			response['fit']['epochs'] = num_epochs
			response['fit']['lr'] = lr
			response['fit_one_cycle']['max_lr'] = lr
			response['fit_one_cycle']['moms'] = str(moms)

		getattr(DashTrain, response['training']['type'])(response, learn)
		emit('Trained model; completed step 3.')
	else:
		emit('Skipping step 3 because there is no GPU.')

	emit('STEP 4 (optional): Visualizing the attributions.')
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
	emit('Completed visualization; completed step 4.')

	print('STEP 5: Saving the model.')
	# save_path = save_dir / save_name
	# if not save_dir.exists():
	# 	save_dir.mkdir()
	# learn.export(save_path)
	emit('Saved the model; completed step 5. Congratulations!')
	print('(Not actually saving right now; uncomment the relevant lines if needed.)')
	print('Load the model again with the following code:', end='\n\n')
	print(f'\tlearn = load_learner(path={save_dir!r}, file={save_name!r})', end='\n\n')
	emit('-' * 50)
	print('Now we need to add production-serving.')
    return jsonify(res)


# @socketio.on('connect')
# def talk_to_me():
#     emit('after connect',  {'data':'Lets dance'})


if __name__ == "__main__":
    socketio.run(app)