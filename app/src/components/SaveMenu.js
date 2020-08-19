import React, { useState, useEffect } from 'react';
import JsonEditor from './Jsoneditor';
import CButton from './Button';
import { update_type, update_data } from '../redux/actions/data';
import { connect } from 'react-redux';
import { withRouter } from 'react-router';
import Button from 'react-bootstrap/esm/Button';

function SaveMenu({history, data, task, setTask, setData}) {
    const [, forceUpdate] = useState();

    useEffect(() => {
        if(history.action === "PUSH"){
            forceUpdate({});
        }
    }, [])

    const handClick = () => {
        history.goBack();
    }

    const handleChange = (update_data) => {
        setData(update_data)
    }

    return(
        <div style={{width: "80%"}} className={'model-menu'}>
            <JsonEditor 
            onChange={handleChange} 
            data={data["save"]} 
            Title={"Saved Model"} />
        </div>
    )
}

const stateToProps = (state) => {
    return {
        "task": state.payload.task,
        "data": state.payload.data
    }
}

const dispatchToProps = (Dispatch) => {
    return {
        "setData": (val) => Dispatch(update_data(val)),
        "setTask": (val) => Dispatch(update_type(val))
    }
}

export default withRouter(connect(stateToProps, dispatchToProps)(SaveMenu));