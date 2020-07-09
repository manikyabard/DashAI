from vision.databunch import DashVisionDatabunch
import json
import torch.optim
import path

path = path.Path('./')
with open('./data/response.json') as f:
			response = json.load(f)
databunch = DashVisionDatabunch.create_vision_databunch(response)
print('created databunch')
print(databunch)
print('done')