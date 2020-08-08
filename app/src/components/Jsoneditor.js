import React, { useState } from 'react';
// import NormalInput from './NormalInput';
import Icon from 'react-icons-kit';
import {chevronLeft} from 'react-icons-kit/fa/chevronLeft'
import JsonFiled from './JsonFiled';
const sample = {
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
}

const GetRenderer = ({sample, header, state}) => {
    const [expand, setExpand] = useState(state == undefined ? true:false);
    return <div className={"json-div"}>
        <div 
        style={{
            backgroundColor: expand ? "#a1a1a1c9":""
        }}
        className={"collapsable"} onClick={() => setExpand(!expand)}>
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
            return <JsonFiled label={ele} key={index} placeholder={" " + sample[ele] + " "}/>
        }
        else return <GetRenderer sample={sample[ele]} header={ele} state={false}/>
    })} </div>
    
    </div>
}

const JsonEditor = ({Title, data}) => {
    
    
    return(
        <GetRenderer sample={data} header={Title}/>
    )
}


export default JsonEditor;