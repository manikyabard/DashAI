import React, { useState } from 'react';
import NormalInput from './NormalInput';
import Icon from 'react-icons-kit';
import {chevronLeft} from 'react-icons-kit/fa/chevronLeft'
const sample = {
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

const GetRenderer = ({sample, header}) => {
    const [expand, setExpand] = useState(false);
    return <div className={"json-div"}>
        <div className={"collapsable"} onClick={() => setExpand(!expand)}>
        <h3>{header}</h3>
        <Icon style={{
            position: 'absolute',
            top: "26%",
            right: "20px",
            transition: 'all ease-in-out .2s',
            transform: expand ? 'rotate(-90deg)':''
        }} icon={chevronLeft} size={20} />
        </div>
        <div className={"content"} style={{display: expand ? "block":"none"}}>
    { Object.keys(sample).map((ele, index) => {
        if(sample[ele] === null || typeof(sample[ele]) !== "object"){
            return <NormalInput key={index} placeholder={ele + " [default: " + sample[ele] + "]"}/>
        }
        else return <GetRenderer sample={sample[ele]} header={ele} />
    })} </div>
    
    </div>
}

const JsonEditor = () => {
    
    
    return(
        <GetRenderer sample={sample} header={"Home"}/>
    )
}


export default JsonEditor;