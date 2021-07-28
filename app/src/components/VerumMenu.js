import React, { useState, useEffect } from 'react';
import JsonEditor from './Jsoneditor';
import Button from './Button';
import { update_type, update_data } from '../redux/actions/data';
import { connect } from 'react-redux';
import { withRouter } from 'react-router';

function VerumMenu({history, data, task, setTask, setData}) {
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
        setData({
            ...data,
            "verum": update_data
        })
    }

    
    return(
        <div className={'model-menu'}>
            <div className={"header-main"}>
                <Button onClick={handClick} label={"Back"} type={"back"}/>
            </div>
            <JsonEditor 
            onChange={handleChange} 
            data={data.verum} 
            Title={"Verum : AI based Model Optimizer"} />
        </div>
    )
}

const stateToProps = (state) => {
    return {
        "task": state.payload.data.task,
        "data": state.payload
    }
}

const dispatchToProps = (Dispatch) => {
    return {
        "setData": (val) => Dispatch(update_data(val)),
        "setTask": (val) => Dispatch(update_type(val))
    }
}

export default withRouter(connect(stateToProps, dispatchToProps)(VerumMenu));