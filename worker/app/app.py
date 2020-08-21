from flask import Flask, render_template, request, jsonify
import json
import torch
import fastai
import os
from pathlib import Path
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
granddir = os.path.dirname(parentdir) 
sys.path.insert(0,granddir) 

from text.learner import DashTextLearner
from tabular.learner import DashTabularLearner
from vision.learner import DashVisionLearner
from collabfilter.learner import DashCollabLearner
from core.train import DashTrain
from insights.DashInsights import DashInsights
from captum.insights.attr_vis import AttributionVisualizer
from flask_socketio import SocketIO, emit


app = Flask(__name__, template_folder="template")
socketio = SocketIO(app, cors_allowed_origins="*") # socket object to setup websocket connection
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
    
    response = json.loads(request.data)
    application = response['task']
    save_dir = Path(response['save']['save_dir'])
    save_name = Path(response['save']['save_name'])
    learner_class = learner_class_map[application]
    learn = getattr(learner_class, f'create_{application}_learner')(response)
    print(print('-'*10, 'Created learner', '-'*10))

    

    return jsonify(res)

@app.route("/train", methods=['POST'])
def train():
    res = {
        "status": "COMPLETE",
        "message": "",
        "payload": []
    }
    
    response = json.loads(request.data)
    train = json.loads(response["train"])
    verum = json.loads(response["verum"])
 
    with open('./data/verum.json', 'w') as outfile:
        json.dump(verum, outfile)
 
    emit('STEP 2 (optional): Optimizing the hyper-parameters.')
    step_2 = False  # If step 2 done, then later use returned hyper-parameters.
    # Else, use default or mentioned hyper-parameters.
    try:
        import ax
        print(ax.__version__)
        from verum.DashVerum import DashVerum
        step_2 = True
        with open('./data/verum.json') as f:
            response = json.load(f)
        # print(response)
        verum = DashVerum(response, learn)
        learn, metric, lr, num_epochs, moms = verum.veritize()
        print('Hyper-parameters optimized; completed step 2.')
    except ImportError:
        print('Skipping step 2 as the module `ax` is not installed.')
 
    with open('./data/train.json', 'w') as outfile:
        json.dump(train, outfile)
     
    
    return jsonify(res)
     
    
@socketio.on('training',namespace='/home')
def training_worker():
    emit('training', 'STEP 3: Training the model.')
    if torch.cuda.is_available():
        with open('./data/train.json') as f:
            response = json.load(f)
        if step_2:
            response['fit']['epochs'] = num_epochs
            response['fit']['lr'] = lr
            response['fit_one_cycle']['max_lr'] = lr
            response['fit_one_cycle']['moms'] = str(moms)

        getattr(DashTrain, response['training']['type'])(response, learn)
        emit('training', 'Trained model; completed step 3.')
    else:
        emit('training', 'Skipping step 3 because there is no GPU.')

    emit('training', 'STEP 4 (optional): Visualizing the attributions.')
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
    emit('training', 'Completed visualization; completed step 4.')

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


# @socketio.on('connect')
# def talk_to_me():
#     print('after connect',  {'data':'Lets dance'})


if __name__ == "__main__":
    socketio.run(app, port=5001)