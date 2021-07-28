import React from 'react';
import '../pages/New.css';
import {folderOpen} from 'react-icons-kit/fa/folderOpen'
import { Icon } from 'react-icons-kit';

const FolderInput = ({placeholder}) => {
    return(
        <div className={'option'}>
            <div className={'o-col1'}>
                <input placeholder={placeholder} /> 
            </div>
            <div className={'o-col2'}>
                <Icon icon={folderOpen} size={20} />
            </div>
        </div>
    )
}

export default FolderInput;