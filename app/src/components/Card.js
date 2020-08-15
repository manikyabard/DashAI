import React, { useEffect } from 'react';
import './Card.css';
import { connect } from 'react-redux';
import { update_type } from '../redux/actions/data';

const Card = ({task, history, setTask}) => {

    const handleChange = () => {
        setTask(task);
        history.push("/task");
    }

    return(
        <div onClick={handleChange} className="main">
            <h3>{task}</h3>
        </div>
    )
}

const stateToProps = (state) => {
    return {
        "o_task": state.payload.task,
    }
}

const dispatchToProps = (Dispatch) => {
    return {
        "setTask": (val) => Dispatch(update_type(val))
    }
}

export default connect(stateToProps, dispatchToProps)(Card);