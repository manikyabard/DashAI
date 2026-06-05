import importlib
import sys
import types
import unittest


class FakePreds:
    def __init__(self, values):
        self.values = values

    def __add__(self, other):
        return FakePreds([left + right for left, right in zip(self.values, other.values)])

    def __truediv__(self, value):
        return FakePreds([item / value for item in self.values])


class FakeProbs(FakePreds):
    def __add__(self, other):
        return FakeProbs([left + right for left, right in zip(self.values, other.values)])

    def __truediv__(self, value):
        return FakeProbs([item / value for item in self.values])

    def argmax(self):
        return self.values.index(max(self.values))


class FakeLearner:
    def __init__(self, preds, probs=None):
        self.preds = preds
        self.probs = probs
        self.data = types.SimpleNamespace(classes=['negative', 'positive'])
        self.model = object()
        self.recorder = object()
        self.calls = []

    def fit_one_cycle(self, *args, **kwargs):
        self.calls.append(('fit_one_cycle', args, kwargs))

    def get_preds(self, *args, **kwargs):
        return self.preds, 'labels'

    def predict(self, item, *args, **kwargs):
        return None, None, self.probs


def install_fastai_stubs():
    torch = types.ModuleType('torch')
    torch.__path__ = []
    torch_optim = types.ModuleType('torch.optim')
    sys.modules['torch'] = torch
    sys.modules['torch.optim'] = torch_optim

    fastai = types.ModuleType('fastai')
    fastai_text = types.ModuleType('fastai.text')
    fastai_text.__all__ = ['Path', 'AWD_LSTM', 'language_model_learner', 'text_classifier_learner']
    fastai_text.Path = lambda value: value
    fastai_text.AWD_LSTM = object()
    fastai_text.language_model_learner = lambda *args, **kwargs: None
    fastai_text.text_classifier_learner = lambda *args, **kwargs: None

    transformer = types.ModuleType('fastai.text.models.transformer')
    transformer.init_transformer = lambda *args, **kwargs: None

    sys.modules['fastai'] = fastai
    sys.modules['fastai.text'] = fastai_text
    sys.modules['fastai.text.models'] = types.ModuleType('fastai.text.models')
    sys.modules['fastai.text.models.transformer'] = transformer

    databunch = types.ModuleType('text.databunch')
    databunch.DashTextDatabunch = object()
    sys.modules['text.databunch'] = databunch


class BidirectionalTextLearnerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        install_fastai_stubs()
        cls.learner_module = importlib.import_module('text.learner')

    def test_get_preds_averages_forward_and_backward_outputs(self):
        forward = FakeLearner(FakePreds([0.2, 0.8]))
        backward = FakeLearner(FakePreds([0.6, 0.4]))

        learner = self.learner_module.BidirectionalTextLearner(forward, backward)
        averaged, labels = learner.get_preds()

        self.assertEqual(averaged.values, [0.4, 0.6000000000000001])
        self.assertEqual(labels, 'labels')

    def test_fit_one_cycle_trains_both_directional_learners(self):
        forward = FakeLearner(FakePreds([0.2, 0.8]))
        backward = FakeLearner(FakePreds([0.6, 0.4]))

        learner = self.learner_module.BidirectionalTextLearner(forward, backward)
        learner.fit_one_cycle(2, max_lr=0.01)

        self.assertEqual(forward.calls, [('fit_one_cycle', (2,), {'max_lr': 0.01})])
        self.assertEqual(backward.calls, [('fit_one_cycle', (2,), {'max_lr': 0.01})])

    def test_predict_averages_probabilities_and_returns_best_class(self):
        forward = FakeLearner(FakePreds([0.2, 0.8]), FakeProbs([0.3, 0.7]))
        backward = FakeLearner(FakePreds([0.6, 0.4]), FakeProbs([0.9, 0.1]))

        learner = self.learner_module.BidirectionalTextLearner(forward, backward)
        class_name, class_idx, averaged = learner.predict('A sample review')

        self.assertEqual(class_name, 'negative')
        self.assertEqual(class_idx, 0)
        self.assertEqual(averaged.values, [0.6, 0.39999999999999997])


if __name__ == '__main__':
    unittest.main()
