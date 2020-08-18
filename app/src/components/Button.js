import React from 'react';
import { connect } from 'react-redux';

import { Icon } from 'react-icons-kit';
import {androidArrowBack} from 'react-icons-kit/ionicons/androidArrowBack'
import {androidClose} from 'react-icons-kit/ionicons/androidClose'
const CButton = ({label, onClick, task, type}) => {
    return(
        <div onClick={() => onClick(label)} className={(task === label)?"btn-active":"btn-main"}>
            <Icon icon={type === "back" ? androidArrowBack:androidClose} size={30}/>
        </div>
    )
}


const stateToProps = (state) => {
    return {
        "task": state.payload.task,
    }
}

const dispatchToProps = (Dispatch) => {
    return {
        
    }
}

export default connect(stateToProps, dispatchToProps)(CButton);