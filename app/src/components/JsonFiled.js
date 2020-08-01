import React from 'react';


const JsonFiled = ({placeholder, label}) => {
    return(
        <div style={{padding: "10px"}}>
            <div className={'option'}>
                <div className={'o-col2-json'}>
                    <p>{label}</p>
                </div>
                <div className={'o-col1'}>
                <input placeholder={placeholder} /> 
                </div>
            </div>
        </div>
        )
    }
    
    export default JsonFiled;