import React, { useState, useEffect } from 'react';
import JsonEditor from './Jsoneditor';
import CButton from './Button';
import { update_type, update_data } from '../redux/actions/data';
import { connect } from 'react-redux';
import { withRouter } from 'react-router';
import Button from 'react-bootstrap/esm/Button';

function ModelMenu({history, data, task, setTask, setData}) {
    const [, forceUpdate] = useState();

    useEffect(() => {
        if(history.action === "PUSH"){
            forceUpdate({});
        }
    }, [])

    const handClick = (label) => {
        if (label === "Back")
            history.goBack();
        else history.push("/core")
    }

    const handleChange = (json) => {
        // console.log(update_data)
        setData({
            ...data,
            "data": {
                ...data["data"],
                [task]: {
                    ...data["data"][task],
                    "input": json
                }
            }
        })
    }

    return(
        <div className={'model-menu'}>
            <div className={"header-main"}>
                <CButton onClick={handClick} label={"Back"} type={"back"}/>
                <div style={{
                    position: "absolute",
                    right: "70px",
                    top: "0px"
                }}>
                    <CButton onClick={handClick} label={"Forward"} type={"core"}/>
                </div>
            </div>
            <JsonEditor 
            onChange={handleChange} 
            data={data["data"][task]["input"]} 
            Title={"Input"} />
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

export default withRouter(connect(stateToProps, dispatchToProps)(ModelMenu));