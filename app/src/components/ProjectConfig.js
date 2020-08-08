import React from 'react';
import '../pages/New.css';
import FolderInput from './FolderInput';
import NormalInput from './NormalInput';
import { JsonEditor as Editor } from 'jsoneditor-react';
import 'jsoneditor-react/es/editor.min.css';
const ProjectConfig = () => {
    return(
        <div className="getting-started">
            <div className="gs-container">
                <h3>Project Details</h3>
                <FolderInput placeholder={"Folder"} />
                <NormalInput placeholder={"Name"}/>
                
                <hr />

                <h3>Data</h3>
                <Editor
                    value={{
                        "bs": 8,
                        "val_bs": null,
                        "device": null,
                        "no_check": false,
                        "num_workers": 16,
                        "validation": {
                            "method": "none",
                            "by_rand_pct": {
                                "valid_pct": 0.2,
                                "seed": null
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
                                "col": null
                            }
                        }
                    }
                }
                
                    // onChange={this.handleChange}
                />
            </div>
        </div>
    )
}

export default ProjectConfig;