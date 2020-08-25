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
// import console_output from '../assets/result.txt';
// import userHome from 'user-home';

//const socket = io('http://localhost:5001/home');


import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';


const override = `
  display: block;
  margin: 0 auto;
  border-color: red;
`;


const TunnelPage = ({visibility, setVisibility, res}) => {

    const [generated, setGenerated] = useState(false);
    const [train, setTrain] = useState(false);
    const [result, setResult] = useState("");
    const [home, setHome] = useState("");
    const [serverRes, setServerRes] = useState("UNSET");
    const [captum, setCaptum] = useState(false);
    const handlePop = () => {
        fetch("http://localhost:5001/stop", {
                method: 'GET',
                "Access-Control-Allow-Origin": "*",
            }).then(response => response.json())
            .then(data => {
                setVisibility(false);
                setGenerated(false);
                setTrain(false);
            })   
    }
    
    const  showFile = async (home) => {
        var p = new XMLHttpRequest();
        p.open('GET', home + '/.dashai/result.txt', false);
        p.send(null);
        setResult(p.responseText)
        // p.onreadystatechange = () => {
        //     if (p.readyState === 4) {
                
        //     }
        // }
        // e.preventDefault()
        // const reader = new FileReader()
        // reader.onload = async (e) => { 
        //   const text = (e.target.result)
        //   setResult(text);
        // //   alert(text)
        // };
        // // console.log(e.target.files)
        // reader.readAsText(e.target.files[0])
      } 

    useEffect(() => {

            // handleChangeFile(console_output)
            console.log(visibility);
            fetch("http://localhost:5001/gethome", {
                method: 'GET',
                "Access-Control-Allow-Origin": "*",
            }).then((response, err) =>
                response.json())
            .then(data =>
                setHome(data.payload))
        }
    , [])

    useEffect(() => {
        if(visibility && home !== ""){

            console.log(home);
            fetch("http://localhost:5001/generate", {
            method: 'POST',
            "Access-Control-Allow-Origin": "*",
            body: JSON.stringify(res.data)
        }).then(response => response.json())
        .then(data => {
            if(data.status === "COMPLETE"){
                setServerRes("GENERATED")
                fetch("http://localhost:5001/train", {
                method: 'POST',
                "Access-Control-Allow-Origin": "*",
                body: JSON.stringify({
                    "train": res.train,
                    "verum": res.verum,
                    "data": res.data
                })
                }).then(response => response.json())
                .then(data => {
                    setServerRes("TRAINING")
                })
                
            }   
        })
        }
    }, [home, visibility])

    useEffect(() => {
        if(serverRes === "GENERATED"){
            setGenerated(true);
        }else if(serverRes === "NO_SERVER"){

        }else if(serverRes === "TRAINED"){
            toast.success('Model is Trained', {
                position: "bottom-right",
                autoClose: 5000,
                hideProgressBar: false,
                closeOnClick: true,
                pauseOnHover: true,
                draggable: true,
                progress: undefined,
            });
            setCaptum(true);
        }
    }, [serverRes])

    useEffect(() => {
        if(result.includes("COMPLETE")){
            setServerRes("TRAINED");
        }
    }, [result])

    const handleTrain = () => {
        fetch("http://localhost:5001/start", {
            method: 'GET',
            "Access-Control-Allow-Origin": "*",
        }).then(response => response.json())
        .then(data => {
            setTrain(true);
        })

        setInterval(() => {
            showFile(home);
        }, 1000);
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
                <div id={"captum"} style={{
                    display: captum ? "block":"none",
                }} className={'btn-gp'}>
                <a target="_blank" href={"http://127.0.0.1:5003"}>Open DashInsights</a>
                </div>
                <div style={{
                    display: train ? "block": "none"
                }} className={"console"}>
                    <div style={{
                        position: 'absolute',
                        left: "50%",
                        transform: "translateX(-40%)",
                        bottom: "80px"
                    }}>
                        <CButton onClick={() => showFile(home)} label={"reload"} type={"close"}/>
                    </div>
                    
                    <pre>{result}</pre>
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
            <ToastContainer
            position="bottom-right"
            autoClose={5000}
            hideProgressBar={false}
            newestOnTop={false}
            closeOnClick
            rtl={false}
            pauseOnFocusLoss
            draggable
            pauseOnHover
            />
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