import React from 'react';
// import '../pages/New.css';
// import {folderOpen} from 'react-icons-kit/fa/folderOpen'
// import { Icon } from 'react-icons-kit';

const NormalInput = ({placeholder}) => {
    return(
        <div style={{padding: "10px"}}>
            <div className={'option'}>
                <div className={'o-col1'}>
                    <input placeholder={placeholder} /> 
                </div>
                <div className={'o-col2'}>
                    <p>JSON</p>
                </div>
            </div>
        </div>
    )
}

export default NormalInput;