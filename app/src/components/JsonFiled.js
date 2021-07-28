import React from 'react';
import { connect } from 'react-redux';



import { JsonEditor as Editor } from 'jsoneditor-react';
import 'jsoneditor-react/es/editor.min.css';
import Ajv from 'ajv';
import { update_data } from '../redux/actions/data';

const ajv = new Ajv({ allErrors: true, verbose: true });

const JsonFiled = ({onChange, data, task, sample, header}) => {
    const handleChange = (json) => {
        // onChange({
        //     ...data,
        //     [task]: {
        //         ...data[task],
        //         [header]: {
        //             ...update_data
        //         }
        //     }
        // });
        console.log(json)
    }

    
    return(
        <div style={{padding: "10px"}}>
            {/*<div className={'option'}>
                <div className={'o-col2-json'}>
                    <p>{label}</p>
                </div>
                <div className={'o-col1'}>
                <input onChange={handleChange} placeholder={placeholder}/> 
    </div>
            </div>*/}
            <Editor
            mode={'tree'}
            ajv={ajv}
            value={sample}
            onChange={onChange}
            />
        </div>
        
        )
}


const stateToProps = (state) => {
    return {
        "task": state.payload.task,
        "data": state.payload
    }
}

const dispatchToProps = (Dispatch) => {
    return {
        
    }
}

export default connect(stateToProps, dispatchToProps)(JsonFiled);