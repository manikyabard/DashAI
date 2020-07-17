from flask import Flask, render_template, request, jsonify
from tabular.learner import DashTabularLearner as tl
from flask_socketio import SocketIO, emit


app = Flask(__name__, template_folder="template")
socketio = SocketIO(app) # socket object to setup websocket connection
learner = None

@app.route("/", methods=['GET'])
def helper():
    return render_template("helper.html")

@app.route("/generate", methods=['POST'])
def generate():
    res = {
        "status": "FAIL",
        "message": "Failed to generate model",
        "payload": []
    }
    type = request.form.get("task")
    if task == "tabular":
        learner = tl.create_tabular_learner(request.data)
        res["status"] = "GENERATE_SUCCESS"
        res["message"] = "Tabular Model Generated"
        res["payload"] = []
    elif task == "vision":
        res["status"] = "FAIL"
        res["message"] = "Under Development"
        res["payload"] = []
    elif task == "text":
        res["status"] = "FAIL"
        res["message"] = "Under Development"
        res["payload"] = []

    return jsonify(res)

@app.route("/train")
def train():
    res = {
        "status": "FAIL",
        "message": "Failed to start traning model",
        "payload": []
    }

    if learner == None:
        res["message"] = "Failed to find a learner"
    else:
        learner.fit_one_cycle(2) # I assume 2 here means epoch


    return jsonify(res)


# @socketio.on('connect')
# def talk_to_me():
#     emit('after connect',  {'data':'Lets dance'})


if __name__ == "__main__":
    socketio.run(app)