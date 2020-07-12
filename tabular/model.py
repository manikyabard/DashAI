from fastai.tabular import *

class DashTabularModel:

	@staticmethod
	def create_tabular_model(databunch, response):

		if response['type'] == 'default':
			out_sz = response['default']['out_sz']
			layers = response['default']['layers']
			emb_drop = response['default']['emb_drop']
			ps = response['default']['ps']
			y_range = response['default']['y_range']
			use_bn = response['default']['use_bn']
			bn_final = response['default']['bn_final']

		model = TabularModel(
			emb_szs=databunch.get_emb_szs(),
			n_cont=len(databunch.cont_names),
			out_sz=out_sz,
			layers=layers,
			emb_drop=emb_drop,
			ps=ps,
			y_range=y_range,
			use_bn=use_bn,
			bn_final=bn_final
		)

		return model
		
	# Input size is probably incorrect. Try running tab_test.py with response_cust_model.json
	@staticmethod
	def create_custom_tabular_model(data, layers, **kwargs):
		emb_szs = data.get_emb_szs()
		input_sz = 1 + len(data.cont_names)
		for x, y in emb_szs:
			input_sz += x - y
		out_sz = data.c
		assert layers[0].in_features == input_sz, f"Input size should be {input_sz} in {layers[0]} ie. the first layer"
		assert layers[-1].out_features == out_sz, f"Output size should be {out_sz} in {layers[-1]} ie. the last layer"
		model = DashCustomTabularModel(emb_szs, len(data.cont_names), out_sz=data.c, layers=layers, **kwargs)
		return model
	

class DashCustomTabularModel(Module):
	"Basic model for tabular data."
	def __init__(self, emb_szs:ListSizes, n_cont:int, out_sz:int, layers, ps:Collection[float]=None,
				 emb_drop:float=0., y_range:OptRange=None, bn_begin:bool=False):
		super().__init__()
		ps = ifnone(ps, [0]*len(layers))
		ps = listify(ps, layers)
		self.embeds = nn.ModuleList([embedding(ni, nf) for ni,nf in emb_szs])
		self.emb_drop = nn.Dropout(emb_drop)
		# self.bn_cont = nn.BatchNorm1d(n_cont)
		if bn_begin: self.bn_cont = nn.BatchNorm1d(n_cont)
		n_emb = sum(e.embedding_dim for e in self.embeds)
		self.n_emb,self.n_cont,self.y_range = n_emb,n_cont,y_range
		self.layers = nn.Sequential(*layers)

	def get_sizes(self, layers, out_sz):
		return [self.n_emb + self.n_cont] + layers + [out_sz]

	def forward(self, x_cat:Tensor, x_cont:Tensor) -> Tensor:   

		if self.n_emb != 0:
			x = [e(x_cat[:,i]) for i,e in enumerate(self.embeds)]
			x = torch.cat(x, 1)
			x = self.emb_drop(x)

		if self.n_cont != 0:
			# x_cont = self.bn_cont(x_cont)
			x = torch.cat([x, x_cont], 1) if self.n_emb != 0 else x_cont

		x = self.layers(x)
		if self.y_range is not None:
			x = (self.y_range[1]-self.y_range[0]) * torch.sigmoid(x) + self.y_range[0]
		return x

	