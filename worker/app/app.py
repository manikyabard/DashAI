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
from flask_cors import CORS

# import sys
# sys.stdout = open('result.txt', 'w')

# sys.stdout.flush()

# with open('result.txt', 'w+') as f:
#     f.write("worked here")

app = Flask(__name__, template_folder="template")
app.config['SECRET_KEY'] = 'some-super-secret-key'
app.config['DEFAULT_PARSERS'] = [
    'flask.ext.api.parsers.JSONParser',
    'flask.ext.api.parsers.URLEncodedParser',
    'flask.ext.api.parsers.FormParser',
    'flask.ext.api.parsers.MultiPartParser'
]
cors = CORS(app,resources={r"/*":{"origins":"*"}})
socketio = SocketIO(app, cors_allowed_origins="*", ping_timeout=5) # socket object to setup websocket connection
# learner = None
# learn = metric = lr = num_epochs = moms = None
ready_to_train = False
application = ""
save_dir = ""
save_name = ""
path = Path('./')
learner_class_map = {
    'collab': DashCollabLearner,
    'tabular': DashTabularLearner,
    'text': DashTextLearner,
    'vision': DashVisionLearner
}

step_2 = False  # If step 2 done, then later use returned hyper-parameters.
# Else, use default or mentioned hyper-parameters.

@app.route("/", methods=['GET'])
def helper():
    print("Started")
    return render_template("helper.html")

@app.route("/generate", methods=['POST'])
def generate():
    res = {
        "status": "COMPLETE",
        "message": "",
        "payload": []
    }
    global learn
    global application
    global save_dir
    global save_name
    response = json.loads(request.data)
    application = response['task']
    save_dir = Path(response['save']['save_dir'])
    save_name = Path(response['save']['save_name'])
    learner_class = learner_class_map[application]
    learn = getattr(learner_class, f'create_{application}_learner')(response)
    print('-'*10, 'Created learner', '-'*10)

    return jsonify(res)

@app.route("/train", methods=['POST'])
def train():
    res = {
        "status": "COMPLETE",
        "message": "",
        "payload": []
    }
    global learn
    print("line 76", learn)
    response = json.loads(request.data)
    print(response)
    train = response["train"]
    verum = response["verum"]
 
    with open('./data/verum.json', 'w') as outfile:
        json.dump(verum, outfile)
 
    print('STEP 2 (optional): Optimizing the hyper-parameters.')
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
    
    if res["status"] == "SUCCESS":
        global ready_to_train
        ready_to_train = True
    return jsonify(res)
     

@app.route("/start", methods=['GET'])
def start():
    res = {
        "status": "COMPLETE",
        "message": "",
        "payload": []
    }
    training_worker()
    return jsonify(res)


@socketio.on('connect',  namespace='/home')
def connected():
    emit('connect', {"msg": "STEP 3: Training the model."}, namespace="/home")

@socketio.on('training',namespace='/home')
def training_worker():
    global application
    global save_dir
    global save_name
    emit('connect',{"msg": "STEP 3: Training the model."}, namespace="/home", broadcast=True)
    # if torch.cuda.is_available():
    if True:
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


# @socketio.on('connect')
# def talk_to_me():
#     print('after connect',  {'data':'Lets dance'})


if __name__ == "__main__":
    # app.run(debug=True, host='0.0.0.0')
    socketio.run(app, port=5001, debug=True)