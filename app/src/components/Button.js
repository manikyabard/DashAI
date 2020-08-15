import React from 'react';
import { connect } from 'react-redux';
const Button = ({label, onClick, task}) => {
    return(
        <div onClick={() => onClick(label)} className={(task === label)?"btn-active":"btn-main"}>
            <h3>
                {label}
            </h3>
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

export default connect(stateToProps, dispatchToProps)(Button);