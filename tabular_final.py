from fastai.tabular import * 
path = untar_data(URLs.ADULT_SAMPLE)
import torch.optim

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
      },

      

    "loss": {
      'type': 'pre-defined',

      'pre-defined': {
        'func': CrossEntropyFlat
      },

      'custom': {
        'fname': None,
        'func': None
      }  
    },

    "model": {
      'type': 'default',
      'default': {
        'out_sz': None,						# int
        'layers': None,						# Collection[int]
        'emb_drop': 0.0,
        'ps': None,							# Collection[float]
        'y_range': None,						# tuple(float, float)
        'use_bn': True,
        'bn_final': False
      }
    },


    "optimizer": {
      'available_opts':['SGD','RMSProp','Adam','AdamW','Adadelta','Adagrad','SparseAdam','Adamax','ASGD'],
      'chosen_opt':'Adamax',
      'arguments': {
          'SGD':
          {
              'lr':0,                          # learning rate
              'momentum':0.0,                  # momentum factor (default: 0)       
              'weight_decay':0.0,              # weight decay (L2 penalty) (default: 0)
              'dampening':0.0,                  # dampening for momentum (default: 0)
              'nesterov':False                 # enables Nesterov momentum (default: False)
          },
          'RMSProp':
          {
              'lr':0.01,                      # learning rate (default: 1e-2)
              'momentum':0.0,                 # momentum factor (default: 0)
              'alpha':0.99,                   # smoothing constant (default: 0.99)
              'eps':1e-08,                    # term added to the denominator to improve numerical stability (default: 1e-8)
              'centered':False,               # if True, compute the centered RMSProp, the gradient is normalized by an estimation of its variance
              'weight_decay':0                # weight decay (L2 penalty) (default: 0)
          },
          'Adam':
          {
              'lr':0.001,                     # learning rate (default: 1e-3)
              'momentum':0.9,                 # default
              'alpha':0.999,                   # coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999))
              'eps':1e-08,                    # term added to the denominator to improve numerical stability (default: 1e-8)
              'weight_decay':0.0,            # weight decay (L2 penalty) (default: 0)
              'amsgrad':False                 # whether to use the AMSGrad variant of this algorithm from the paper On the Convergence of Adam and Beyond (default: False)
          },
          'AdamW':
          {
              'lr':0.001,                     # learning rate (default: 1e-3)
              'momentum':0.9,                 # default
              'alpha':0.999,                   # coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999))
              'eps':1e-08,                    # term added to the denominator to improve numerical stability (default: 1e-8)
              'weight_decay':0.01,            # weight decay (L2 penalty) (default: 0.01)
              'amsgrad':False                 # whether to use the AMSGrad variant of this algorithm from the paper On the Convergence of Adam and Beyond (default: False)
          },
          'Adadelta':
          {
              'lr':1,                          # coefficient that scale delta before it is applied to the parameters (default: 1.0)
              'rho':0.9,                     # coefficient used for computing a running average of squared gradients (default: 0.9)
              'eps':1e-06,                    # term added to the denominator to improve numerical stability (default: 1e-6)
              'weight_decay':0                # weight decay (L2 penalty) (default: 0)
          },
          'Adagrad':
          {
              'lr':0.01,                       # learning rate (default: 1e-2)
              'lr_decay':0.0,                  # learning rate decay (default: 0)
              'eps':1e-10,                     # term added to the denominator to improve numerical stability (default: 1e-10)
              'weight_decay':0                # weight decay (L2 penalty) (default: 0)
          },
          'SparseAdam':
          {
              'lr':0.001,                     # learning rate (default: 1e-3)
              'momentum':0.9,                 # default
              'alpha':0.999,                  # coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999))
              'eps':1e-08                     # term added to the denominator to improve numerical stability (default: 1e-8)
          },
          'Adamax':
          {
              'lr':0.002,                     # learning rate (default: 0.002)
              'momentum':0.9,                 # default
              'alpha':0.999,                   # coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999))
              'eps':1e-08,                    # term added to the denominator to improve numerical stability (default: 1e-8)
              'weight_decay':0.01            # weight decay (L2 penalty) (default: 0.01)
              
          },
          'ASGD':
          {
              'lr':0.01,                     # learning rate (default: 0.002)
              'lambd':0.0001,                 # decay term (default: 1e-4)
              'momentum':0.9,                 # default
              'alpha':0.75,                  # power for eta update (default: 0.75)
              't0':1000000.0,                    # point at which to start averaging (default: 1e6)
              'weight_decay':0.0,            # weight decay (L2 penalty) (default: 0.0)
              
          }
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
            src = src.split_by_rand_pct(
                valid_pct=response['validation']['rand_pct']['valid_pct'],
                seed=response['validation']['rand_pct']['seed']
            )

        if response['validation']['method'] == 'subsets':
            src = src.split_subsets(
                train_size=response['validation']['subsets']['train_size'],
                valid_size=response['validation']['subsets']['valid_size'],
                seed=response['validation']['subsets']['seed']
            )

        if response['validation']['method'] == 'files':       #TODO: test it out
            src = src.split_by_files(valid_name=response['validation']['files']['valid_names'])

        if response['validation']['method'] == 'fname_file':
            src = src.split_by_fname_file(
                fname=response['validation']['fname_files']['fname'],
                path=response['validation']['fname_files']['path']
            )

        if response['validation']['method'] == 'folder':
            src = src.split_by_folder(
                train=response['validation']['folder']['train'],
                valid=response['validation']['folder']['valid']
            )

        if response['validation']['method'] == 'idx':
            valid_idx = range(len(df)-response['validation']['idx']['valid_idx'], len(df))
            src = src.split_by_idx(valid_idx)

        if response['validation']['method'] == 'idxs':
            src = src.split_by_idxs(
                train_idx = response['validation']['idxs']['train_idx'],
                valid_idx = response['validation']['idxs']['valid_idx']
            )

        if response['validation']['method'] == 'list':
            src = src.split_by_list(
                train = response['validation']['list']['train'],
                valid = response['validation']['list']['valid']
            )

        if response['validation']['method'] == 'valid_func':
            funcfile = response['validation']['valid_func']['fname']
            import funcfile as fname
            func = fname.response['validation']['valid_func']['func']
            src = src.split_by_valid_func(func)

        if response['validation']['method'] == 'from_df':
            src = src.split_from_df(col=response['validation']['from_df']['col'])

        # Step 3: Label the inputs
        if response['label']['method'] == 'from_df':      #TODO test it out
            if not response['label']['from_df']['classes']:
                src = src.label_from_df(cols=response['dep_var'])
            else:
                src.label_from_df(
                    cols=response['label']['from_df']['cols'],
                    label_cls=response['label']['from_df']['label_cls'],
                    one_hot=response['label']['from_df']['one_hot'],
                    classes=response['label']['from_df']['classes']
                )

        if response['label']['method'] == 'empty':
            src = src.label_empty()

        if response['label']['method'] == 'const':
            src = src.label_const(
                const=response['label']['const']['const'],
                label_cls=response['label']['const']['label_cls']
            )

        if response['label']['method'] == 'from_func':
            funcfile = response['label']['from_func']['fname']
            import funcfile as fname
            func = fname.response['label']['from_func']['func']
            src = src.label_from_func(func)

        if response['label']['method'] == 're':
            src = src.label_from_re(
                pat=response['label']['re']['pat'],
                full_path=response['label']['re']['full_path']
            )

        # Optional: add test
        if response['test_df']:
            src.add_test(TabularList.from_df(response['test_df'],
                         cat_names=cat_names,
                         cont_names=cont_names,
                         processor=src.train.x.processor))

        # Step 4: Convert to databunch
        return src.databunch(
            path=path,
            bs=response['bs'],
            val_bs=response['val_bs'],
            num_workers=response['num_workers'],
            device=response['device'],
            no_check=response['no_check']
        )

    except Exception as e:
        print('Exception:', e)

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

def create_tabular_loss(response):

	if response['type'] == 'pre-defined':
		loss_func = response['pre-defined']['func']

	else:
		#import response['custom']['fname']
		func = fname.response['custom']['func']
		loss_func = FlattenedLoss(func)

	return loss_func

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


def create_optimize(response):
  try:
    if response['chosen_opt']=='SGD':
      arg=response['arguments']['SGD']
      opt_func = partial(optim.SGD,lr=arg['lr'],momentum=arg['momentum'],dampening=arg['dampening'], nesterov=arg['nesterov'],weight_decay=arg['weight_decay'])


    if response['chosen_opt']=='RMSProp':
      arg=response['arguments']['RMSProp']
      opt_func = partial(optim.RMSprop,lr=arg['lr'],eps=arg['eps'],momentum=arg['momentum'],alpha=arg['alpha'],weight_decay=arg['weight_decay'],centered=arg['centered'])


    if response['chosen_opt']=='Adam':
      arg=response['arguments']['Adam']
      opt_func = partial(optim.Adam,lr=arg['lr'],betas=(arg['momentum'],arg['alpha']),eps=arg['eps'],weight_decay=arg['weight_decay'],amsgrad=arg['amsgrad'])


    if response['chosen_opt']=='AdamW':
      arg=response['arguments']['AdamW']
      opt_func = partial(optim.AdamW,lr=arg['lr'],betas=(arg['momentum'],arg['alpha']),eps=arg['eps'],amsgrad=arg['amsgrad'],weight_decay=arg['weight_decay'])


    if response['chosen_opt']=='Adadelta':
      arg=response['arguments']['Adadelta']
      opt_func = partial(optim.Adadelta,lr=arg['lr'],eps=arg['eps'],weight_decay=arg['weight_decay'],rho=arg['rho'])

    if response['chosen_opt']=='Adagrad':
      arg=response['arguments']['Adagrad']
      opt_func = partial(optim.Adagrad,lr=arg['lr'],lr_decay=arg['lr_decay'],eps=arg['eps'],weight_decay=arg['weight_decay'])

    if response['chosen_opt']=='SparseAdam':
      arg=response['arguments']['SparseAdam']
      opt_func = partial(optim.SparseAdam,lr=arg['lr'],betas=(arg['momentum'],arg['alpha']),eps=arg['eps'])

    if response['chosen_opt']=='Adamax':
      arg=response['arguments']['Adamax']
      opt_func = partial(optim.Adamax,lr=arg['lr'],betas=(arg['momentum'],arg['alpha']),eps=arg['eps'],weight_decay=arg['weight_decay'])

    if response['chosen_opt']=='ASGD':
      arg=response['arguments']['ASGD']
      opt_func = partial(optim.ASGD,lr=arg['lr'],momentum=arg['momentum'],alpha=arg['alpha'],lambd=arg['lambd'],t0=arg['t0'],weight_decay=arg['weight_decay'])
    
    return opt_func
  except Exception as e:
    print(e)

def func():
  databunch=create_tabular_databunch(response)
  metrics = create_tabular_metric(response)
  #loss=create_tabular_loss(response['loss'])    getting an error while training when i passed this in the learner with loss_func=loss
  #mod=create_tabular_model(databunch,response['model'])  not sure where to pass this in the learner
  opt=create_optimize(response['optimizer'])
  learn = tabular_learner(databunch, layers=[200,100], emb_szs={'native-country': 10},metrics=metrics,opt_func=opt)
  return learn

learn=func()
learn.fit_one_cycle(1, 1e-2)