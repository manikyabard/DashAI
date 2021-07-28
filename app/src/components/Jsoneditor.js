import React, { useState, useEffect } from 'react';
// import NormalInput from './NormalInput';
import Icon from 'react-icons-kit';
import {chevronLeft} from 'react-icons-kit/fa/chevronLeft'
import JsonFiled from './JsonFiled';


import { JsonEditor as Editor } from 'jsoneditor-react';
import 'jsoneditor-react/es/editor.min.css';
import Ajv from 'ajv';

const ajv = new Ajv({ allErrors: true, verbose: true });


const GetRenderer = ({sample, header, state, onChange}) => {
    const [expand, setExpand] = useState(state === undefined ? true:false)

    // return <Editor
    //         ajv={ajv}
    //         value={sample}
    //         onChange={onChange}
    //         />

    
    return <div className={"json-div"}>
        <div 
        style={{
            backgroundColor: expand ? "#eeeeee":"#eeeeee"
        }}
        className={"collapsable"} onClick={() => setExpand(!expand)}>
        <h2 style={{padding: "20px"}}>{header}</h2>
        <Icon style={{
            position: 'absolute',
            top: "26%",
            right: "20px",
            transition: 'all ease-in-out .2s',
            transform: expand ? 'rotate(-90deg)':''
        }} icon={chevronLeft} size={20} />
        </div>
        <div className={"content"} style={{display: expand ? "block":"none"}}>
            <JsonFiled 
            // key={index}
            sample={sample}
            onChange={onChange} 
        />
        </div>
    </div>
}

const JsonEditor = ({Title, data, onChange}) => {

    return(
        <GetRenderer onChange={onChange} sample={data} header={Title}/>
    )
}


export default JsonEditor;