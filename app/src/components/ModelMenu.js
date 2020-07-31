import React from 'react';
import {folderOpen} from 'react-icons-kit/fa/folderOpen'
import { Icon } from 'react-icons-kit';
import NormalInput from './NormalInput';
import JsonEditor from './Jsoneditor';
// import { JsonEditor as Editor, JsonEditor } from 'jsoneditor-react';
// import 'jsoneditor-react/es/editor.min.css';


function ModelMenu(props) {
    return(
        <div className={'model-menu'}>
            <JsonEditor />
        </div>
    )
}

export default ModelMenu;