from fastai.collab import *
#Not working!!
class DashCollabModel:

	@staticmethod
	def create_collab_model(databunch, response):

		if response['type'] == 'default':
			n_factor=response['default']['n_factor']
			use_nn=response['default']['use_nn']
			layers = response['default']['layers']
			emb_drop = response['default']['emb_drop']
			ps = response['default']['ps']
			y_range = tuple(response['default']['y_range'])
			use_bn = response['default']['use_bn']
			bn_final = response['default']['bn_final']
		print(y_range)
		emb_szs = databunch.get_emb_szs()
		u,m = databunch.train_ds.x.classes.values()
		print(u,m)
		if use_nn:
			model = EmbeddingNN(emb_szs=emb_szs,layers=layers,ps=ps,emb_drop=emb_drop,y_range=y_range,use_bn=use_bn,bn_final=bn_final)
			return model
		else:
			model = EmbeddingDotBias(n_factor,len(u),len(m),y_range=y_range)
			return model
			
	'''	
	# Input size is probably incorrect. Try running tab_test.py with response_cust_model.json
	@staticmethod
	def create_custom_collab_model(data, layers, **kwargs):
		emb_szs = data.get_emb_szs()
		input_sz = len(data.cont_names)
		for x, y in emb_szs:
			input_sz += y
		out_sz = data.c
		assert layers[0].in_features == input_sz, f"Input size should be {input_sz} in {layers[0]} ie. the first layer"
		assert layers[-1].out_features == out_sz, f"Output size should be {out_sz} in {layers[-1]} ie. the last layer"
		model = DashCustomCollabModel(emb_szs, len(data.cont_names), out_sz=data.c, layers=layers, **kwargs)
		return model
	'''
	
#No idea how to implement this for collaborative filtering
class DashCustomCollabModel(nn.Module):
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

	