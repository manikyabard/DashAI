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
            'label_cls': None,                  #or MultiCategoryList
            'items': None,
            'label_delim': None,
            'one_hot': True,
            'classes': None                     #list of classes
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
    'num_workers': 16
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
                     
databunch = create_tabular_databunch(response)
