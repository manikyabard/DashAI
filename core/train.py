import fastai
from fastai.basic_train import *
from fastai.callbacks import *

class DashTrain():
    ''' Contains methods related to pre-training options, training,
     and post-training.
    '''
    @staticmethod
    def fit(response, learn: Learner):
        response["training"]["fit"]["lr"] = eval(response["training"]["fit"]["lr"])
        learn.fit(**response["training"]["fit"])

    @staticmethod
    def fit_one_cycle(response, learn: Learner):
        response["training"]["fit_one_cycle"]["max_lr"] = eval(response["training"]["fit_one_cycle"]["max_lr"])
        response["training"]["fit_one_cycle"]["moms"] = eval(response["training"]["fit_one_cycle"]["moms"])
        learn.fit_one_cycle(**response["training"]["fit_one_cycle"])

    @staticmethod
    def lr_find(response, learn: Learner):
        learn.lr_find(**response["pre-training options"]["lr_find"])

    @staticmethod
    def to_fp16(response, learn: Learner):
        learn.to_fp16(**response["pre-training options"]["lr_find"])

    @staticmethod
    def to_fp32(learn):
        learn.to_fp32()
    
    @staticmethod
    def plot_metrics(learn):
        learn.recorder.plot_metrics()

    @staticmethod
    def plot_losses(learn):
        learn.recorder.plot_losses()

    
    


    

    

