import React, { useState, useEffect } from 'react';
import JsonEditor from './Jsoneditor';
import CButton from './Button';
import { update_type, update_data } from '../redux/actions/data';
import { connect } from 'react-redux';
import { withRouter } from 'react-router';
import Button from 'react-bootstrap/esm/Button';

function TrainModel({history, data, save, setTask, setData}) {
    const [, forceUpdate] = useState();
    useEffect(() => {
        if(history.action === "PUSH"){
            forceUpdate({});
        }
    }, [])

    const handClick = (label) => {
        if(label === "Back")
            history.goBack();   
        else history.push("/verum")
    }

    const handleChange = (update_data) => {
        setData(update_data)
    }

    return(
        <div style={{width: "90%"}} className={'model-menu'}>
            {/*<div className={"header-main"}>
                <CButton onClick={handClick} label={"Back"} type={"back"}/>
                <div style={{
                    position: "absolute",
                    right: "70px",
                    top: "0px"
                }}>
                    <CButton onClick={handClick} label={"Forward"} type={"verum"}/>
                </div>
            </div>*/}
            <JsonEditor 
            onChange={handleChange} 
            data={data} 
            Title={"Training Configuration"} />

            <JsonEditor 
            onChange={handleChange} 
            data={save} 
            Title={"Save Model (Optional)"} />
        </div>
    )
}

const stateToProps = (state) => {
    return {
        "task": state.payload.data.task,
        "data": state.payload.train,
        "save": state.payload.data.save
    }
}

const dispatchToProps = (Dispatch) => {
    return {
        "setData": (val) => Dispatch(update_data(val)),
        "setTask": (val) => Dispatch(update_type(val))
    }
}

export default withRouter(connect(stateToProps, dispatchToProps)(TrainModel));