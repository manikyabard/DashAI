from vision.databunch import DashVisionDatabunch
import json
import torch.optim
import path

path = path.Path('./')
with open('./data/response-coco.json') as f:
			response = json.load(f)
databunch = DashVisionDatabunch.create_vision_databunch(response)
print('created databunch')
print(databunch)
# databunch.show_batch(rows=2, figsize=(6,6))
print('done')