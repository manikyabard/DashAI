from fastai.tabular import * 

path = untar_data(URLs.ADULT_SAMPLE)

response = {
    'csv_name': 'adult.csv',

    'FillMissing': {
        'fill_strategy': FillStrategy.MEDIAN,   #Default
        'add_col': True,                        #Default
        'fill_val': 0.0                         #Default
    },

    'Categorify': True,                         #Default
    'Normalize': True,                          #Default

    'Datetime': {
        'cols': [],                             #col names with datatime type
        'cyclic': False                         #Default
    },

    'dep_var': 'salary',
    'cat_names': ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country'],
    'cont_names': ['fnlwgt', 'capital-gain', 'age', 'education-num', 'capital-loss', 'hours-per-week'],
    'valid_set_size': 2000,
    'test_df': None,                            #Default  (put name of test csv)
    'bs': 64,                                   #Default
    'val_bs': None,                             #Default
    'device': None,

    'validation': {
        'method': 'idx',                   #[None, rand_pct, subsets, files, fname_files, folder, idx, idxs, list, valid_func, from_df]
        'rand_pct': {
            'valid_pct': 0.2,
            'seed': None
        },
        'idx': {
            'valid_idx': 20
        },
        'subsets': {
            'train_size': 0.08,
            'valid_size': 0.2,
            'seed': None
        },
        'files': {
            'valid_names': None                 #ItemList
        },
        'fname_file': {
            'fname': None,                      #PathorStr
            'path': None                        #PathorStr
        },
        'folder': {
            'train': 'train',                   #PathOrStr
            'valid': 'train'                    #PathOrStr
        },
        'idxs': {
            'train_idx': None,
            'valid_idx': None
        },
        'list': {
            'train': None,
            'valid': None
        },
        'valid_func': {
            'fname': None,                      #File name with function defined.
            'func': None                        #Name of function.
        },
        'from_df': {
            'col': None                         #IntsorStrs.    col with True for valid
        }
    },

    'label': {
        'method': 'from_df',                    # [empty, from_df, const, from_folder]
        'from_df': {
            'cols': None,
            'label_cls': None,                  # or MultiCategoryList/FloatList etc.
            'items': None,
            'label_delim': None,
            'one_hot': True,
            'classes': None                     # list of classes
        },
        'const': {
            'const': 0,
            'label_cls': None                   # can maybe add more args
        },
        'from_func': {
            'fname': None,
            'func': None
        },
        'from_re': {
            'pat': None,
            'full_path': False
        }
    },

    'no_check': False,
    'num_workers': 16,

    # Options for metric: 
    #  [accuracy, accuracy_thresh, top_k_accuracy, dice, error_rate, mean_squared_error, mean_absolute_error,
    #    mean_squared_logarithmic_error, exp_rmspe, root_mean_squared_error, fbeta, explained_variance, r2_score,
    #    Precision, Recall, FBeta, ExplainedVariance, MatthewsCorreff, KappaScore,
    #    MultiLabelFbeta, auc_roc_score, roc_curve, AUROC]


    'metric': {
        'methods': ['accuracy', 'error_rate', 'Precision'],
        'accuracy_thresh': {
            'thresh': 0.5,                      # Default
            'sigmoid': True                     # Default
        },
        'top_k_accuracy': {
            'k': 5                              # Default
        },
        'dice': {
            'iou': False,                       # Default
            'eps': 1e-08,                       # Default
        },
        'fbeta': {
            'thresh': 0.2,                      # Default
            'beta': 2.0,                        # Default
            'eps': 1e-09,                       # Default
            'sigmoid': True                     # Default
        },
        'Precision': {
            'average': 'binary',                # Default, options: `binary`, `micro`, `macro`, `weighted` or None
            'pos_label': 1,                     # Default
            'eps': 1e-09                        # Default
        },
        'Recall': {
            'average': 'binary',                # Default, options: `binary`, `micro`, `macro`, `weighted` or None
            'pos_label': 1,                     # Default
            'eps': 1e-09                        # Default
        },
        'FBeta': {
            'average': 'binary',                # Default, options: `binary`, `micro`, `macro`, `weighted` or None
            'pos_label': 1,                     # Default
            'eps': 1e-09,                       # Default
            'beta': 2.0
        },
        'KappaScore':{
            'weights': None                     # Default, options: None, `linear`, or `quadratic`
        },
        'MultiLabelFbeta': {
            'beta': 2,                          # Default
            'eps': 1e-15,                       # Default
            'thresh': 0.3,                      # Default
            'sigmoid': True,                    # Default
            'average': 'micro'                  # Default, options: None, ‘binary’, ‘micro’, ‘macro’, ‘samples’, ‘weighted’
        }
      }

}

''' Options for metric: 
      [accuracy, accuracy_thresh, top_k_accuracy, dice, error_rate, mean_squared_error, mean_absolute_error,
        mean_squared_logarithmic_error, exp_rmspe, root_mean_squared_error, fbeta, explained_variance, r2_score,
        Precision, Recall, FBeta, ExplainedVariance, MatthewsCorreff, KappaScore,
        MultiLabelFbeta, auc_roc_score, roc_curve, AUROC]
'''
def create_tabular_metric(response):
  try:
    if response['metric']['methods'] == None:
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
        ))

      if kind == 'top_k_accuracy':
        kinds.append(partial(
            top_k_accuracy,
            k = response['metric']['top_k_accuracy']['k']
        ))

      if kind == 'dice':
        kinds.append(partial(
            dice,
            iou = response['metric']['dice']['iou'],
            eps = response['metric']['dice']['eps']
        ))

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
        ))
      
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
    print(e)


databunch = create_tabular_databunch(response)
metrics = create_tabular_metric(response)
learn = tabular_learner(databunch, layers=[200,100], emb_szs={'native-country': 10}, metrics=metrics)
