from fastai.text import *
from .databunch import DashTextDatabunch
# from .metric import DashTabularMetric
# from .model import DashTabularModel
# from .loss import DashTabularLoss
# from .optimizer import DashTabularOptimizer

import fastai
import torch.optim
from fastai.text.models.transformer import init_transformer


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

        if response["arch"] == 'AWD_LSTM':
            config = response['configs']['AWD_LSTM']

        if response["arch"] == 'Transformer':
            config = response['configs']['Transformer']
            config['act'] = eval(config['act'])
            config['init'] = eval(config['init'])

        if response["arch"] == 'TransformerXL':
            config = response['configs']['TransformerXL']
            config['act'] = eval(config['act'])
            config['init'] = eval(config['init'])
        
        arch = eval(response['arch'])
        return language_model_learner(data, arch, config = config, **response['extra'])


    @staticmethod
    def create_text_classifier_learner_default(data, response):

        if response["arch"] == 'AWD_LSTM':
            config = response['configs']['AWD_LSTM']

        if response["arch"] == 'Transformer':
            config = response['configs']['Transformer']
            config['act'] = eval(config['act'])
            config['init'] = eval(config['init'])

        if response["arch"] == 'TransformerXL':
            config = response['configs']['TransformerXL']
            config['act'] = eval(config['act'])
            config['init'] = eval(config['init'])

        arch = eval(response['arch'])
        return text_classifier_learner(data, arch, config = config, **response['extra'])

