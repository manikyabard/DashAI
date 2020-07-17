from fastai.text import *
from .databunch import DashTextDatabunch
# from .metric import DashTabularMetric
# from .model import DashTabularModel
# from .loss import DashTabularLoss
# from .optimizer import DashTabularOptimizer

import fastai
import torch.optim


class DashTextLearner:

    @staticmethod
    def create_text_learner(response):
        path = Path('./')
        data = DashTextDatabunch.create_text_databunch(response)

        if response['text']['model']['type'] == "classifier":
            if response['text']['model']['classifier']['method'] == "default":
                return DashTextLearner.create_text_classifier_learner_default(data, response['text']['model']['classifier']['default'])
        
        if response['text']['model']['type'] == "language_model":
            if response['text']['model']['language_model']['method'] == "default":
                return DashTextLearner.create_text_lm_learner_default(data, response['text']['model']['language_model']['default'])

    
    @staticmethod
    def create_text_lm_learner_default(data, response):
        arch = eval(response['arch'])
        return language_model_learner(data, arch, **response['extra'])


    @staticmethod
    def create_text_classifier_learner_default(data, response):
        arch = eval(response['arch'])
        return text_classifier_learner(data, arch, **response['extra'])

