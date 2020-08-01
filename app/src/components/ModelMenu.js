import React from 'react';
import {folderOpen} from 'react-icons-kit/fa/folderOpen'
import { Icon } from 'react-icons-kit';
import NormalInput from './NormalInput';
import JsonEditor from './Jsoneditor';
import DropDown from './DropDown';
import Button from './Button';
// import { JsonEditor as Editor, JsonEditor } from 'jsoneditor-react';
// import 'jsoneditor-react/es/editor.min.css';

const data = {
    "default": {
        "out_sz": 3,
        "emb_drop": 0,
        "ps": null,
        "y_range": null,
        "use_bn": true,
        "bn_final": false,
        "layers": [64,64],
    },
    "custom": {
        "layers": ["nn.Linear(4, 5)", "nn.ReLU()", "nn.Linear(5, 3)"],
        "extra_args": {
            "bn_begin": false
        }
    }
}


function ModelMenu(props) {
    const handClick = () => {

    }
    return(
        <div className={'model-menu'}>
            <div style={{
                backgroundColor: "#a1a1a1c9"
            }} className={'btn-gp'}>
                <Button label={"Text"}/>
                <Button label={"Vision"}/>
            </div>
            <div style={{
                backgroundColor: "#a1a1a1c9"
            }} className={'btn-gp'}>
                <Button onClick={handClick} label={"Tabular"}/>
                <Button onClick={handClick} label={"CT"}/>
            </div>
            <JsonEditor data={data} Title={"Configuration"} />
        </div>
    )
}

export default ModelMenu;