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
        func: CrossEntropyFlat
      }

      'custom': {
        fname: None,
        func: None
      }  
    },

    "model": {
      'type': 'default',
      'default': {
        'out_sz' = None,						# int
        'layers' = None,						# Collection[int]
        'emb_drop' = 0.0,
        'ps' = None,							# Collection[float]
        'y_range' = None,						# tuple(float, float)
        'use_bn' = True,
        'bn_final' = False
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
