
{
    "task": "tabular",
    "core": {
        "data": {
            "bs": 64,       #Default
            "val_bs": null, #Default
            "device": null, #Default
            "no_check": false, #Default
            "num_workers": 16,
            "validation": {
                "method": "none",  # [split_none, split_by_rand_pct, split_subsets, split_by_files, split_by_fname_file, split_by_folder, split_by_idx, split_by_idxs, split_by_list, split_by_valid_func, split_from_df]
                "rand_pct": {
                    "valid_pct": 0.2,  #Default
                    "seed": null   #Default
                },
                "idx": {
                    "csv_name": null,
                    "valid_idx": 20
                },
                "subsets": {
                    "train_size": 0.08,
                    "valid_size": 0.2,
                    "seed": null
                },
                "files": {
                    "valid_names": null
                },
                "fname_file": {
                    "fname": null,
                    "path": null
                },
                "folder": {
                    "train": "train",
                    "valid": "train"
                },
                "idxs": {
                    "train_idx": null,
                    "valid_idx": null
                },
                "list": {
                    "train": null,
                    "valid": null
                },
                "valid_func": {
                    "fname": null,
                    "func": null
                },
                "from_df": {
                    "col": null  #Default is 2
                }
            },
    
            "label": {
                "method": "from_df",  # [label_empty, label_from_df, label_const, label_from_folder, label_from_func, label_from_re]
                "from_df": {
                    "default":true,
                    "cols": 1,  #Default
                    "label_cls": null,   #Default   options: null, FloatList, CategoryList, MultiCategoryList, EmptyLabelList
                    "items": null,
                    "label_delim": null,
                    "one_hot": false,
                    "classes": null
                },
                "const": {
                    "const": 0,  #Default
                    "label_cls": null  #Default
                },
                "from_func": {
                    "fname": null,
                    "func": null
                },
                "from_re": {
                    "pat": null,
                    "full_path": false
                },
                "from_folder": {
                    "label_cls": null
                }
            }
        },
        "metric": {
            
            #  Available options: [accuracy, accuracy_thresh, top_k_accuracy, dice, error_rate, mean_squared_error, mean_absolute_error,
            #   mean_squared_logarithmic_error, exp_rmspe, root_mean_squared_error, fbeta, explained_variance, r2_score, Precision, Recall,
            #   FBeta, ExplainedVariance, MatthewsCorreff, KappaScore, MultiLabelFbeta, auc_roc_score, roc_curve, AUROC]
            
            "methods": [            
                "accuracy",
                "error_rate",
                "Precision"
            ],
            "accuracy_thresh": {
                "thresh": 0.5,      #Default
                "sigmoid": true      #Default
            },
            "top_k_accuracy": {
                "k": 5      #Default
            },
            "dice": {
                "iou": false,      #Default
                "eps": 1e-8      #Default
            },
            "fbeta": {
                "thresh": 0.2,      #Default
                "beta": 2,      #Default
                "eps": 1e-9,      #Default
                "sigmoid": true      #Default
            },
            "Precision": {
                "average": "binary",      #Default
                "pos_label": 1,      #Default
                "eps": 1e-9      #Default
            },
            "Recall": {
                "average": "binary",      #Default
                "pos_label": 1,      #Default
                "eps": 1e-9      #Default
            },
            "FBeta": {
                "average": "binary",      #Default
                "pos_label": 1,      #Default
                "eps": 1e-9,      #Default
                "beta": 2      #Default
            },
            "KappaScore": {
                "weights": null
            },
            "MultiLabelFbeta": {
                "beta": 2,      #Default
                "eps": 1e-15,      #Default
                "thresh": 0.3,      #Default
                "sigmoid": true,      #Default
                "average": "micro"      #Default
            }
        },
        "loss": {
            "type": "pre-defined",
            "pre-defined": {
                "func": "MSELossFlat" # BCEFlat, BCEWithLogitsFlat, CrossEntropyFlat, MSELossFlat, NoopLoss, WassersteinLoss
            },
            "custom": {
                "fname": null,
                "func": null
            }
        },
        "optimizer": {
            # "available_opts": [
            #     "SGD",
            #     "RMSProp",
            #     "Adam",
            #     "AdamW",
            #     "Adadelta",
            #     "Adagrad",
            #     "SparseAdam",
            #     "Adamax",
            #     "ASGD"
            # ],
            "chosen_opt": "ASGD",
            "arguments": {
                "SGD": {
                    "lr": 0,
                    "momentum": 0,
                    "weight_decay": 0,
                    "dampening": 0,
                    "nesterov": false
                },
                "RMSProp": {
                    "lr": 0.01,
                    "momentum": 0,
                    "alpha": 0.99,
                    "eps": 1e-8,
                    "centered": false,
                    "weight_decay": 0
                },
                "Adam": {
                    "lr": 0.001,
                    "momentum": 0.9,
                    "alpha": 0.999,
                    "eps": 1e-8,
                    "weight_decay": 0,
                    "amsgrad": false
                },
                "AdamW": {
                    "lr": 0.001,
                    "momentum": 0.9,
                    "alpha": 0.999,
                    "eps": 1e-8,
                    "weight_decay": 0.01,
                    "amsgrad": false
                },
                "Adadelta": {
                    "lr": 1,
                    "rho": 0.9,
                    "eps": 0.000001,
                    "weight_decay": 0
                },
                "Adagrad": {
                    "lr": 0.01,
                    "lr_decay": 0,
                    "eps": 1e-10,
                    "weight_decay": 0
                },
                "SparseAdam": {
                    "lr": 0.001,
                    "momentum": 0.9,
                    "alpha": 0.999,
                    "eps": 1e-8
                },
                "Adamax": {
                    "lr": 0.002,
                    "momentum": 0.9,
                    "alpha": 0.999,
                    "eps": 1e-8,
                    "weight_decay": 0.01
                },
                "ASGD": {
                    "lr": 0.01,
                    "lambd": 0.0001,
                    "alpha": 0.75,
                    "t0": 1000000,
                    "weight_decay": 0
                }
            }
        }
    },
    "tabular": {
        "input": {
            "csv_name": "./data/hello.csv",
            "dep_var": "columnt",
            "cat_names": [
                "column1"
            ],
            "cont_names": [
                "column2"
            ],
            "test_df": {
                "has_test": false,
                "csv_name": null
            }
        },
        "transform": {
            "FillMissing": {
                "fill_strategy": "MEDIAN",  # MEDIAN, COMMON, CONSTANT
                "add_col": true,
                "fill_val": 0 #Filled with this if CONSTANT
            },
            "Categorify": true,
            "Normalize": true,
            "Datetime": {
                "cols": [],
                "cyclic": false #Bool
            }
        },
        "model": {
            "type": "default",  #Default, Custom
            "default": {
                "out_sz": null,
                "layers": null,
                "emb_drop": 0,
                "ps": null,
                "y_range": null,
                "use_bn": true,
                "bn_final": false
            },
            "custom": {
                "layers": [],
                "extra_args": {
                    "bn_begin": false
                }
            }
        }
    },
    
    "test_df": null,
    

    "vision": {
        "subtask": "object-detection",
        "input": {
            "method": "from_folder",
            "from_folder": {
                "path": "data/coco_tiny",
                "extensions": null,
                "recurse": true,
                "exclude": null,
                "include": null,
                "processor": null,
                "presort": false
            },
            "from_csv": {
                "csv_name": null,
                "path":null,
                "cols": 0,
                "delimiter": null,
                "header": "infer",
                "processor": null
            }
        },
        "classification-single-label": {},
        "classification-multi-label": {},
        "regression": {},
        "segmentation": {},
        "gan": {
            "noise_sz": 100
        },
        "object-detection": {},
        "points": {},
        "transform":
        {
            "size":24,
            "data_aug":["basic_transforms","zoom_crop","manual"],
            "chosen_data_aug":"manual",
            "basic_transforms":
            {
                "do_flip":true, 
                "flip_vert":false, 
                "max_rotate":10.0, 
                "max_zoom":1, 
                "max_lighting":0.8, 
                "max_warp":0.2, 
                "p_affine":0.75, 
                "p_lighting":0.75
            },
            "zoom_crop":
            {
                "scale":[0.75,2],
                "do_rand":true,
                "p":1.0
            },
            "manual":
            {
              "brightness":
              {
                  "change":0.5
              },
              "contrast":
              {
                  "scale":1.0
              },
              "crop":
              {
                  "size":300,
                  "row_pct":0.5,
                  "col_pct":0.5
              },
              "crop_pad":
              {
                  "size":300,
                  "padding_mode":"reflection",
                  "row_pct":0.5,
                  "col_pct":0.5
              },
              "dihedral":
              {
                  "k":0
              },
              "dihedral_affine":
              {
                  "k":0
              },
              "flip_lr":
              {},
              "flip_affine":
              {},
              "jitter":
              {
                  "magnitude":0.0
              },
              "pad":
              {
                  "padding":50,
                  "available_modes":["zeros", "border", "reflection"],
                  "mode":"reflection"
              },
              "rotate":
              {
                  "degrees":0.0
              },
              "rgb_randomize":
              {
                  "channels":["Red", "Green", "Blue"],
                  "chosen_channels":"Red",
                  "thresh":[0.2,0.595,0.99],
                  "chosen_thresh":0.2
              },
              "skew":
              {
                  "direction":0,
                  "magnitude":0,
                  "invert":false
              },
              "squish":
              {
                  "scale":1.0,
                  "row_pct":0.5,
                  "col_pct":0.5
              },
              "symmetric_wrap":
              {
                  "magnitude":[-0.2,0.2]
              },
              "tilt":
              {
                  "direction":0,
                  "magnitude":0
              },
              "zoom":
              {
                  "scale":1.0,
                  "row_pct":0.5,
                  "col_pct":0.5
              },
              "cutout":
              {
                  "n_holes":1,
                  "length":40
              }
            }
        }
    }
}