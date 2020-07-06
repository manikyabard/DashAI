from fastai.tabular import * 

path = untar_data(URLs.ADULT_SAMPLE)

import torch.optim

optimizer = {
    'available_opts':['SGD','RMSProp','Adam','AdamW','Adadelta','Adagrad','SparseAdam','Adamax','ASGD'],
    'chosen_opt':'Adamax',
    'arguments':
    {
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
    
    
opt=create_optimize(optimizer)

learn = tabular_learner(databunch, layers=[200,100], emb_szs={'native-country': 10}, opt_func=opt)

learn.fit_one_cycle(1, 1e-2)