import React, { useEffect, useState } from 'react';
import CButton from './Button';
import { connect } from 'react-redux';
import BarLoader from "react-spinners/BarLoader";
import  Button from 'react-bootstrap/Button';
import ClimbingBoxLoader from "react-spinners/ClimbingBoxLoader";
import Selectr from 'jsoneditor/dist/jsoneditor-minimalist';
import SaveMenu from './SaveMenu';
import TrainMenu from './TrainMenu';
import io from 'socket.io-client';
import console_output from '../assets/result.txt';
const socket = io('http://localhost:5001/home');

const override = `
  display: block;
  margin: 0 auto;
  border-color: red;
`;



const TunnelPage = ({visibility, setVisibility, res}) => {

    const [generated, setGenerated] = useState(false);
    const [train, setTrain] = useState(false);
    const handlePop = () => {
        setVisibility(false);
    }

    useEffect(() => {
        if(visibility){
            fetch("http://localhost:5001/generate", {
            method: 'POST',
            body: JSON.stringify(res.data)
        }).then(response => response.json())
        .then(data => {
            console.log(data)
            fetch("http://localhost:5001/train", {
            method: 'POST',
            body: JSON.stringify({
                "train": res.train,
                "verum": res.verum
            })
        }).then(response => response.json())
        .then(data => {
            console.log(data);
        })
        })
        setTimeout(() => {
            setGenerated(true);
        }, 3000);
        
    } else {
        setGenerated(false);
        setTrain(false);
    }

       
    }, [visibility])

    const handleTrain = () => {
        setTrain(true);
        fetch("http://localhost:5001/start", {
            method: 'GET',
        }).then(response => response.json())
        .then(data => {
            console.log(data)
        })
        socket.on('training', data => {
            console.log(data);
        });
    }

    return(
        <div 
        style={{
            display: visibility ? "flex":"none"
        }}
        className={"pop-main"}>
            <div className={"training-page"}>
                <div className={"tunnel-train-btn"}>
                    <CButton onClick={handlePop} label={"Cancel"} type={"close"}/>
                </div>
                <div style={{
                    display: train ? "block": "none"
                }} className={"console"}>
                    <p>{console_output}</p>
                </div>
                <div style={{
                    display: !generated ? "flex": "none"
                }} className={"training"}>
                    <BarLoader
                        css={override}
                        size={150}
                        color={"#2987bc"}
                        loading={!generated}
                    />
                    
                    <h3>Generating Model</h3>
                </div>
                <div style={{
                    display: !train && generated ? "flex": "none"
                }} className={"training"}>
                    <TrainMenu />
                </div>
                <div style={{
                    display: !train && generated ? "block": "none"
                }} className={"header"}>
                    <div className={'btn-gp'}>
                    <Button size="lg" onClick={handleTrain} variant="light">Train Model</Button>
                    </div>

                    
                </div>
            </div>
        </div>
    )
}


const stateToProps = (state) => {
    return {
        "res": state.payload,
    }
}

const dispatchToProps = (Dispatch) => {
    return {
        
    }
}


export default connect(stateToProps, dispatchToProps)(TunnelPage);