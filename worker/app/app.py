from flask import Flask, render_template, request, jsonify
import json
import torch
from torch import Tensor
import fastai
import os
from pathlib import Path
import os,sys,inspect
import multiprocessing 
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
# from flask_socketio import SocketIO, emit
from flask_cors import CORS

import logging
import socket
import threading
from time import sleep
from typing import Optional
from captum.log import log_usage

visualizer = None
port = None

# import sys
# sys.stdout = open('result.txt', 'w')

# sys.stdout.flush()

# with open('result.txt', 'w+') as f:
#     f.write("worked here")

app = Flask(
    __name__, static_folder="../../captum/insights/attr_vis/frontend/build/static", 
    template_folder="../../captum/insights/attr_vis/frontend/build"
    )
app.config['SECRET_KEY'] = 'some-super-secret-key'
app.config['DEFAULT_PARSERS'] = [
    'flask.ext.api.parsers.JSONParser',
    'flask.ext.api.parsers.URLEncodedParser',
    'flask.ext.api.parsers.FormParser',
    'flask.ext.api.parsers.MultiPartParser'
]
cors = CORS(app,resources={r"/*":{"origins":"*"}})
#socketio = SocketIO(app, cors_allowed_origins="*", ping_timeout=5) # socket object to setup websocket connection
# learner = None
learn = metric = lr = num_epochs = moms = None
ready_to_train = False
application = ""
save_dir = ""
save_name = ""
all_processes = []

path = Path('./')
home = os.path.expanduser('~')
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
    return render_template("index.html")

@app.route('/gethome', methods=['GET'])
def gethome():
    global home
    return jsonify({
        "payload": home
    })

@app.route("/generate", methods=['POST'])
def generate():
    res = {
        "status": "COMPLETE",
        "message": "GENERATED_MODEL",
        "payload": []
    }
    global learn
    global application
    global save_dir
    global save_name
    response = json.loads(request.data)
    with open('./data/response.json', 'w') as outfile:
        json.dump(response, outfile, indent=4)
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
        "message": "TRAINING_MODEL",
        "payload": []
    }
    global learn

    response = json.loads(request.data)
    train = response["train"]
    verum = response["verum"]
    data = response["data"]
    with open('./data/verum.json', 'w') as outfile:
        json.dump(verum, outfile, indent=4)

    with open('./data/train.json', 'w') as outfile:
        json.dump(train, outfile, indent=4) 

    with open('./data/response.json', 'w') as outfile:
        json.dump(data, outfile, indent=4)
 
    print('STEP 2 (optional): Optimizing the hyper-parameters.')
    # try:
    #     import ax
    #     from verum.DashVerum import DashVerum
    #     step_2 = True
    #     with open('./data/verum.json') as f:
    #         response = json.load(f)
    #     verum = DashVerum(response, data, learn)
    #     learn, lr, num_epochs, moms = verum.veritize()

    # except ImportError:


    
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
    
    global all_processes
    training_worker()
    # process = multiprocessing.Process(target=training_worker, args=()) 
    # process.start()
    # process.join()
    # all_processes.append(process)
    return jsonify(res)

@app.route("/stop", methods=['GET'])
def stop():
    global all_processes
    res = {
        "status": "COMPLETE",
        "message": "",
        "payload": []
    }
    
    for process in all_processes:
        process.terminate()
        
    return jsonify(res)
    

#@socketio.on('training',namespace='/home')
def training_worker():
    global application
    global save_dir
    global save_name
    global learn
    global visualizer
    print('connect',{"msg": "STEP 3: Training the model."})
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
        print('training', 'Trained model; completed step 3.')
    else:
        print('training', 'Skipping step 3 because there is no GPU.')

    print('training', 'STEP 4 (optional): Visualizing the attributions.')
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
        # process = multiprocessing.Process(target=visualizer.serve, args=(debug=True)) 
        # process.start()
        # all_processes.append(process)
        print('Completed visualization; completed step 4.')
    else:
        print("Visualization is not possible for this application")

    print('STEP 5: Saving the model.')
    # save_path = save_dir / save_name
    # if not save_dir.exists():
    # 	save_dir.mkdir()
    # learn.export(save_path)
    print('Saved the model; completed step 5. Congratulations!')
    print('Load the model again with the following code:', end='\n\n')
    print(f'\tlearn = load_learner(path={save_dir!r}, file={save_name!r})', end='\n\n')
    print('-' * 50)
    print('COMPLETE')
    global all_processes
    print(all_processes)


# @socketio.on('connect')
# def talk_to_me():

    #socketio.run(app, port=5001, debug=True)
    
    
# ----------------------------------------

#!/usr/bin/env python3


# from flask import Flask, jsonify, render_template, request
# from torch import Tensor



def namedtuple_to_dict(obj):
    if isinstance(obj, Tensor):
        return obj.item()
    if hasattr(obj, "_asdict"):  # detect namedtuple
        return dict(zip(obj._fields, (namedtuple_to_dict(item) for item in obj)))
    elif isinstance(obj, str):  # iterables - strings
        return obj
    elif hasattr(obj, "keys"):  # iterables - mapping
        return dict(
            zip(obj.keys(), (namedtuple_to_dict(item) for item in obj.values()))
        )
    elif hasattr(obj, "__iter__"):  # iterables - sequence
        return type(obj)((namedtuple_to_dict(item) for item in obj))
    else:  # non-iterable cannot contain namedtuples
        return obj


@app.route("/attribute", methods=["POST"])
def attribute():
    # force=True needed for Colab notebooks, which doesn't use the correct
    # Content-Type header when forwarding requests through the Colab proxy
    r = request.get_json(force=True)
    return jsonify(
        namedtuple_to_dict(
            visualizer._calculate_attribution_from_cache(r["instance"], r["labelIndex"])
        )
    )


@app.route("/fetch", methods=["POST"])
def fetch():
    # force=True needed, see comment for "/attribute" route above
    global visualizer
    visualizer._update_config(request.get_json(force=True))
    visualizer_output = visualizer.visualize()
    clean_output = namedtuple_to_dict(visualizer_output)
    return jsonify(clean_output)


@app.route("/init")
def init():
    visualizer
    return jsonify(visualizer.get_insights_config())


@app.route("/")
def index(id=0):
    return render_template("index.html")


def get_free_tcp_port():
    tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp.bind(("", 0))
    addr, port = tcp.getsockname()
    tcp.close()
    return port


def run_app(debug: bool = True):
    app.run(port=port, use_reloader=False, debug=debug)


@log_usage()
def start_server(
    _viz, blocking: bool = False, debug: bool = False, _port: Optional[int] = None
):
    global visualizer
    visualizer = _viz

    global port
    if port is None:
        os.environ["WERKZEUG_RUN_MAIN"] = "true"  # hides starting message
        if not debug:
            log = logging.getLogger("werkzeug")
            log.disabled = True
            app.logger.disabled = True

        # port = _port or get_free_tcp_port()
        port = 5003
        # Start in a new thread to not block notebook execution
        t = threading.Thread(target=run_app, kwargs={"debug": debug})
        t.start()
        sleep(0.01)  # add a short delay to allow server to start up
        print(t, blocking, sep='\n')
        if blocking:
            t.join()

    print(f"\nFetch data and view Captum Insights at http://localhost:{port}/\n")
    return port


if __name__ == "__main__":
    app.run(debug=True, port=5001)