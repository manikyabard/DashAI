from fastai.text import *
from .databunch import DashTextDatabunch

import fastai
import torch.optim
from fastai.text.models.transformer import init_transformer
import copy


class BidirectionalTextLearner:
    def __init__(self, forward_learner, backward_learner):
        self.forward_learner = forward_learner
        self.backward_learner = backward_learner
        self.learners = [forward_learner, backward_learner]
        self.data = forward_learner.data
        self.model = forward_learner.model
        self.recorder = forward_learner.recorder

    def __getattr__(self, attr):
        return getattr(self.forward_learner, attr)

    def _apply_to_learners(self, method, *args, **kwargs):
        for learner in self.learners:
            getattr(learner, method)(*args, **kwargs)
        return self

    def fit(self, *args, **kwargs):
        return self._apply_to_learners('fit', *args, **kwargs)

    def fit_one_cycle(self, *args, **kwargs):
        return self._apply_to_learners('fit_one_cycle', *args, **kwargs)

    def lr_find(self, *args, **kwargs):
        return self._apply_to_learners('lr_find', *args, **kwargs)

    def to_fp16(self, *args, **kwargs):
        return self._apply_to_learners('to_fp16', *args, **kwargs)

    def to_fp32(self, *args, **kwargs):
        return self._apply_to_learners('to_fp32', *args, **kwargs)

    def freeze(self, *args, **kwargs):
        return self._apply_to_learners('freeze', *args, **kwargs)

    def unfreeze(self, *args, **kwargs):
        return self._apply_to_learners('unfreeze', *args, **kwargs)

    def save(self, name, *args, **kwargs):
        self.forward_learner.save('{}_forward'.format(name), *args, **kwargs)
        self.backward_learner.save('{}_backward'.format(name), *args, **kwargs)
        return self

    def export(self, file='export.pkl', *args, **kwargs):
        self.forward_learner.export('forward_{}'.format(file), *args, **kwargs)
        self.backward_learner.export('backward_{}'.format(file), *args, **kwargs)
        return self

    def get_preds(self, *args, **kwargs):
        forward_preds = self.forward_learner.get_preds(*args, **kwargs)
        backward_preds = self.backward_learner.get_preds(*args, **kwargs)
        averaged_preds = (forward_preds[0] + backward_preds[0]) / 2
        return (averaged_preds,) + forward_preds[1:]

    def predict(self, item, *args, **kwargs):
        forward_prediction = self.forward_learner.predict(item, *args, **kwargs)
        backward_prediction = self.backward_learner.predict(item, *args, **kwargs)
        averaged_probs = (forward_prediction[2] + backward_prediction[2]) / 2
        class_idx = averaged_probs.argmax()
        class_name = self.data.classes[int(class_idx)]
        return class_name, class_idx, averaged_probs


class DashTextLearner:

    @staticmethod
    def create_text_learner(response):
        path = Path('./tmp/models/')

        if response['text']['model']['type'] == "classifier":
            if DashTextLearner.is_bidirectional_awdlstm(response):
                return DashTextLearner.create_bidirectional_classifier_learner(response)

            data, data_lm = DashTextDatabunch.create_text_databunch(response)
            c_method = response['text']['model']['classifier']['method']
            l_method = response['text']['model']['language_model']['method']
            response_lm = copy.deepcopy(response)
            response_lm['core']['data']['label']['method'] = 'for_lm'
            lm = DashTextLearner.create_text_lm_learner_default(data_lm, response['text']['model']['language_model']['default'])
            lm.fit_one_cycle(response["text"]["model"]["classifier"]["lm_train_epochs"])
            lm.save_encoder('lm_encoder')
            data.vocab.itos = data_lm.vocab.itos
            clas = DashTextLearner.create_text_classifier_learner_default(data, response['text']['model']['classifier']['default'])
            clas.load_encoder('lm_encoder')
            return clas
        
        if response['text']['model']['type'] == "language_model":
            data = DashTextDatabunch.create_text_databunch(response)
            if response['text']['model']['language_model']['method'] == "default":
                return DashTextLearner.create_text_lm_learner_default(data, response['text']['model']['language_model']['default'])

    @staticmethod
    def is_bidirectional_awdlstm(response):
        classifier = response['text']['model']['classifier']['default']
        if classifier['arch'] != 'AWD_LSTM':
            return False
        config = classifier['configs']['AWD_LSTM']
        return bool(config.get('bidir', False))

    @staticmethod
    def create_bidirectional_classifier_learner(response):
        response_forward = DashTextLearner.create_directional_response(response)
        response_backward = DashTextLearner.create_directional_response(response)

        forward_learner = DashTextLearner.create_directional_classifier_learner(
            response_forward, backwards=False, encoder_name='lm_encoder_forward'
        )
        backward_learner = DashTextLearner.create_directional_classifier_learner(
            response_backward, backwards=True, encoder_name='lm_encoder_backward'
        )

        return BidirectionalTextLearner(forward_learner, backward_learner)

    @staticmethod
    def create_directional_response(response):
        directional_response = copy.deepcopy(response)
        classifier_config = directional_response['text']['model']['classifier']['default']['configs']['AWD_LSTM']
        language_model_config = directional_response['text']['model']['language_model']['default']['configs']['AWD_LSTM']
        classifier_config['bidir'] = False
        language_model_config['bidir'] = False
        return directional_response

    @staticmethod
    def create_directional_classifier_learner(response, backwards, encoder_name):
        data, data_lm = DashTextDatabunch.create_text_databunch(response, backwards=backwards)
        lm = DashTextLearner.create_text_lm_learner_default(data_lm, response['text']['model']['language_model']['default'])
        lm.fit_one_cycle(response["text"]["model"]["classifier"]["lm_train_epochs"])
        lm.save_encoder(encoder_name)
        data.vocab.itos = data_lm.vocab.itos
        clas = DashTextLearner.create_text_classifier_learner_default(data, response['text']['model']['classifier']['default'])
        clas.load_encoder(encoder_name)
        return clas

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
