from fastai.metrics import *
from pathlib import Path
import sys, os

class DashMetrics:
	'''
	Contains methods for specifying metrics to be shown during training
	'''

	@staticmethod
	def create_metric(response):
		'''
		Specifies the metrics to be shown using training using the values given in the response/DashUI.
		'''
		try:
			if response['metric']['methods'] == "None":
				return None

			kinds = list()
			for kind in response['metric']['methods']:
				if kind == 'accuracy':
					kinds.append(accuracy)

				if kind == 'accuracy_thresh':
					kinds.append(partial(
							accuracy_thresh,
							thresh = response['metric']['accuracy_thresh']['thresh'], 
							sigmoid = response['metric']['accuracy_thresh']['sigmoid']
						)
					)

				if kind == 'top_k_accuracy':
					kinds.append(partial(
							top_k_accuracy,
							k = response['metric']['top_k_accuracy']['k']
						)
					)

				if kind == 'dice':
					kinds.append(partial(
							dice,
							iou = response['metric']['dice']['iou'],
							eps = response['metric']['dice']['eps']
						)
					)

				if kind == 'error_rate':
					kinds.append(error_rate)

				if kind == 'mean_squared_error':
					kinds.append(mean_squared_error)

				if kind == 'mean_absolute_error':
					kinds.append(mean_absolute_error)

				if kind == 'mean_squared_logarithmic_error':
					kinds.append(mean_squared_logarithmic_error)

				if kind == 'exp_rmspe':
					kinds.append(exp_rmspe)

				if kind == 'root_mean_squared_error':
					kinds.append(root_mean_squared_error)

				if kind == 'fbeta':
					kinds.append(partial(
							fbeta,
							thresh = response['metric']['fbeta']['thresh'], 
							beta = response['metric']['fbeta']['beta'], 
							eps = response['metric']['fbeta']['eps'],
							sigmoid = response['metric']['accuracy_thresh']['sigmoid']
						)
					)

				if kind == 'explained_variance':
					kinds.append(explained_variance)

				if kind == 'r2_score':
					kinds.append(r2_score)

				if kind == 'Precision':
					precision = Precision(
						average = response['metric']['Precision']['average'],
						pos_label = response['metric']['Precision']['pos_label'],
						eps = response['metric']['Precision']['eps']
					)
					kinds.append(precision)

				if kind == 'Recall':
					recall = Recall(
						average = response['metric']['Recall']['average'],
						pos_label = response['metric']['Recall']['pos_label'],
						eps = response['metric']['Recall']['eps']
					)
					kinds.append(recall)

				if kind == 'FBeta':
					fbetavar = FBeta(
						average = response['metric']['FBeta']['average'],
						pos_label = response['metric']['FBeta']['pos_label'],
						eps = response['metric']['FBeta']['eps'],
						beta = response['metric']['FBeta']['beta']
					)
					kinds.append(fbetavar)

				if kind == 'ExplainedVariance':
					expvar = ExplainedVariance()
					kinds.append(expvar)

				if kind == 'MatthewsCorreff':
					matcoeff = MatthewsCorreff()
					kinds.append(matcoeff)

				if kind == 'KappaScore':
					kap = KappaScore(
						weights = response['metric']['KappaScore']['weights'],
					)
					kinds.append(kap)

				if kind == 'MultiLabelFbeta':
					multilabelfbeta = MultiLabelFbeta(
						beta = response['metric']['MultiLabelFbeta']['beta'],
						eps = response['metric']['MultiLabelFbeta']['eps'],
						thresh = response['metric']['MultiLabelFbeta']['thresh'],
						sigmoid = response['metric']['MultiLabelFbeta']['sigmoid'],
						average = response['metric']['MultiLabelFbeta']['average']
					)
					kinds.append(multilabelfbeta)

				if kind == 'auc_roc_score':
					kinds.append(auc_roc_score)

				if kind == 'roc_curve':
					kinds.append(roc_curve)

				if kind == 'AUROC':
					auroc = AUROC()
					kinds.append(auroc)
			return kinds

		except Exception as e:
			exc_type, exc_obj, exc_tb = sys.exc_info()
			fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
			print(exc_type, fname, exc_tb.tb_lineno)