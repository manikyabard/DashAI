from text.databunch import DashTextDatabunch
import json
import torch.optim
import path

path = path.Path('./')
with open('./data/response_new.json') as f:
			response = json.load(f)
databunch = DashTextDatabunch.create_text_databunch(response)
print('created databunch')
print(databunch)
# databunch.show_batch(rows=2, figsize=(6,6))
print('done')