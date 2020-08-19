import React from 'react';
import { connect } from 'react-redux';

import { Icon } from 'react-icons-kit';
import {androidArrowBack} from 'react-icons-kit/ionicons/androidArrowBack'
import {androidClose} from 'react-icons-kit/ionicons/androidClose'
import {androidArrowForward} from 'react-icons-kit/ionicons/androidArrowForward'
const which = {
    "Back": androidArrowBack,
    "Cancel": androidClose,
    "Forward": androidArrowForward
}

const CButton = ({label, onClick, task, type}) => {
    return(
        <div onClick={() => onClick(label)} className={"btn-main"}>
            <Icon icon={which[label]} size={30}/>
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