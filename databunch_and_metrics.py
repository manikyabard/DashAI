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

def create_tabular_databunch(response):
  try:
    #Step 1: Provide inputs
    df = pd.read_csv(path/f'{response["csv_name"]}')

    procs = list()
    if response['FillMissing']:
      procs.append(partial(FillMissing, fill_strategy=response['FillMissing']['fill_strategy'],
                            fill_val=response['FillMissing']['fill_val'],
                            add_col = response['FillMissing']['add_col']))
    if response['Categorify']:
      procs.append(Categorify)
    if response['Normalize']:
      procs.append(Normalize)
    procs = listify(procs)

    cat_names = response['cat_names']
    cont_names = response['cont_names']
    src = TabularList.from_df(df, path=path, cat_names=cat_names, cont_names=cont_names, procs=procs)

    # Step 2: Split data into train and valid
    if response['validation']['method'] == 'none':
      src = src.split_none()
    if response['validation']['method'] == 'rand_pct':
      src = src.split_by_rand_pct(valid_pct = response['validation']['rand_pct']['valid_pct'], seed=response['validation']['rand_pct']['seed'])
    if response['validation']['method'] == 'subsets':
      src = src.split_subsets(train_size = response['validation']['subsets']['train_size'], valid_size = response['validation']['subsets']['valid_size'], seed=response['validation']['subsets']['seed'])
    #if response['validation']['method'] == 'files':       #TODO: test it out
    #  src = src.split_by_files(valid_name = response['validation']['files']['valid_names'])
    #if response['validation']['method'] == 'fname_file':
    #  src = src.split_by_fname_file(fname = response['validation']['fname_files']['fname'], path=response['validation']['fname_files']['path'])
    #if response['validation']['method'] == 'folder':
    #  src = src.split_by_folder(train=response['validation']['folder']['train'], valid=response['validation']['folder']['valid'])
    if response['validation']['method'] == 'idx':
      valid_idx = range(len(df)-response['validation']['idx']['valid_idx'], len(df))
      src = src.split_by_idx(valid_idx)
    #if response['validation']['method'] == 'idxs':
    #  src = src.split_by_idxs(train_idx = response['validation']['idxs']['train_idx'], valid_idx = response['validation']['idxs']['valid_idx'])
    #if response['validation']['method'] == 'list':
    #  src = src.split_by_list(train = response['validation']['list']['train'], valid = response['validation']['list']['valid'])
    if response['validation']['method'] == 'from_df':
      src = src.split_from_df(col = response['validation']['from_df']['col'])
    

    # Step 3: Label the inputs
    if response['label']['method'] == 'from_df':      #TODO test it out
      src = src.label_from_df(cols=response['dep_var']) if response['label']['from_df']['classes'] is None else src.label_from_df(cols=response['label']['from_df']['cols'], label_cls=response['label']['from_df']['label_cls'], one_hot = response['label']['from_df']['one_hot'], classes = response['label']['from_df']['classes'])
    if response['label']['method'] == 'empty':
      src = src.label_empty()
    if response['label']['method'] == 'const':
      src = src.label_const(const = response['label']['const']['const'], label_cls = response['label']['const']['label_cls'])

    # Optional: add test
    if response['test_df'] is not None: src.add_test(TabularList.from_df(response['test_df'], cat_names=cat_names, cont_names=cont_names,
                                                                    processor = src.train.x.processor))
    # Step 4: Convert to databunch
    return src.databunch(path=path, bs=response['bs'], val_bs=response['val_bs'], num_workers=response['num_workers'], device=response['device'], 
                              no_check=response['no_check'])
  except Exception as e:
    print('Exception:', e)


def create_tabular_metric(response):
  try:
    if response['metric']['methods'] == None:
      return None
      
    kinds = list()
    for kind in response['metric']['methods']:
      if kind == 'accuracy':
        kinds.append(accuracy)

      if kind == 'accuracy_thresh':
        kind.append(partial(
            accuracy_thresh,
            thresh = response['metric']['accuracy_thresh']['thresh'], 
            sigmoid = response['metric']['accuracy_thresh']['sigmoid']
        ))

      if kind == 'top_k_accuracy':
        kind.append(partial(
            top_k_accuracy,
            k = response['metric']['top_k_accuracy']['k']
        ))

      if kind == 'dice':
        kind.append(partial(
            dice,
            iou = response['metric']['dice']['iou'],
            eps = response['metric']['dice']['eps']
        ))

      if kind == 'error_rate':
        kind.append(error_rate)

      if kind == 'mean_squared_error':
        kind.append(mean_squared_error)

      if kind == 'mean_absolute_error':
        kind.append(mean_absolute_error)

      if kind == 'mean_squared_logarithmic_error':
        kind.append(mean_squared_logarithmic_error)

      if kind == 'exp_rmspe':
        kind.append(exp_rmspe)

      if kind == 'root_mean_squared_error':
        kind.append(root_mean_squared_error)
      
      if kind == 'fbeta':
        kind.append(partial(
            fbeta,
            thresh = response['metric']['fbeta']['thresh'], 
            beta = response['metric']['fbeta']['beta'], 
            eps = response['metric']['fbeta']['eps'],
            sigmoid = response['metric']['accuracy_thresh']['sigmoid']
        ))
      
      if kind == 'explained_variance':
        kind.append(explained_variance)

      if kind == 'r2_score':
        kind.append(r2_score)

      if kind == 'Precision':
        precision = Precision(
            average = response['metric']['Precision']['average'],
            pos_label = response['metric']['Precision']['pos_label'],
            eps = response['metric']['Precision']['eps']
        )
        kind.append(precision)

      if kind == 'Recall':
        recall = Recall(
            average = response['metric']['Recall']['average'],
            pos_label = response['metric']['Recall']['pos_label'],
            eps = response['metric']['Recall']['eps']
        )
        kind.append(recall)

      if kind == 'FBeta':
        fbetavar = FBeta(
            average = response['metric']['FBeta']['average'],
            pos_label = response['metric']['FBeta']['pos_label'],
            eps = response['metric']['FBeta']['eps'],
            beta = response['metric']['FBeta']['beta']
        )
        kind.append(fbetavar)
      
      if kind == 'ExplainedVariance':
        expvar = ExplainedVariance()
        kind.append(expvar)
      
      if kind == 'MatthewsCorreff':
        matcoeff = MatthewsCorreff()
        kind.append(matcoeff)
      
      if kind == 'KappaScore':
        kap = KappaScore(
            weights = response['metric']['KappaScore']['weights'],
        )
        kind.append(kap)

      if kind == 'MultiLabelFbeta':
        multilabelfbeta = MultiLabelFbeta(
            beta = response['metric']['MultiLabelFbeta']['beta'],
            eps = response['metric']['MultiLabelFbeta']['eps'],
            thresh = response['metric']['MultiLabelFbeta']['thresh'],
            sigmoid = response['metric']['MultiLabelFbeta']['sigmoid'],
            average = response['metric']['MultiLabelFbeta']['average']
        )
        kind.append(multilabelfbeta)
      
      if kind == 'auc_roc_score':
        kind.append(auc_roc_score)

      if kind == 'roc_curve':
        kind.append(roc_curve)
      
      if kind == 'AUROC':
        auroc = AUROC()
        kind.append(auroc)
  except Exception as e:
    print(e)


databunch = create_tabular_databunch(response)

learn = tabular_learner(databunch, layers=[200,100], emb_szs={'native-country': 10}, metrics=auc_roc_score)
