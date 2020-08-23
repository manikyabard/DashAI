from fastai.vision import *
from .databunch import DashVisionDatabunch


class DashVisionLearner:

	@staticmethod
	def create_vision_learner(response):
		path = Path('./')
		data = DashVisionDatabunch.create_vision_databunch(response)

		if response["vision"]["model"]["type"] == "cnn":
			if response["vision"]["model"]["cnn"]["method"] == "default":
				return DashVisionLearner.create_vision_cnn_learner_default(data, response["vision"]["model"]["cnn"])
		if response["vision"]["model"]["type"] == "unet":
			if response["vision"]["model"]["unet"]["method"] == "default":
				return DashVisionLearner.create_vision_unet_learner_default(data, response["vision"]["model"]["unet"])
		if response["vision"]["model"]["type"] == "custom":
			return DashVisionLearner.create_vision_learner_custom(data, response["vision"]["model"]["custom"])
		if response["vision"]["model"]["type"] == "gan":
			return DashVisionLearner.create_gan_learner_default(data, response["vision"]["model"]["gan"])

	@staticmethod
	def create_vision_cnn_learner_default(data, response):
		if hasattr(models, f"{response['default']['arch']}"):
			base_arch = getattr(models, f"{response['default']['arch']}")
		if response['default']["extra"]['init']: response['default']["extra"]['init'] = eval(
			response['default']["extra"]['init'])
		if response['default']["extra"]['custom_head']: response['default']["extra"]['custom_head'] = eval(
			response['default']["extra"]['custom_head'])
		return cnn_learner(data, base_arch, **response["default"]["extra"])

	@staticmethod
	def create_vision_unet_learner_default(data, response):
		if hasattr(models, f"{response['default']['arch']}"):
			base_arch = getattr(models, f"{response['default']['arch']}")
		return unet_learner(data, base_arch, **response["default"]["extra"])

	# Another(maybe better) way of doing this could be to use create_body function to create custom models.
	@staticmethod
	def create_vision_learner_custom(data, response):
		model = nn.Sequential(*[eval(x) for x in response["layers"]])
		return Learner(data, model, **response["extra"])

	def create_gan_learner_default(data, response):
		generator = basic_generator(**response["generator"])
		critic = basic_critic(**response["critic"])
		switcher = eval(response["switcher"])
		learn = GANLearner.wgan(data, generator, critic, switcher)
		return learn
