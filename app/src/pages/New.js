import React, { useEffect, useState } from 'react';
import './New.css';
import Card from '../components/Card';
import ProjectConfig from '../components/ProjectConfig';
// import FloatingBtn from '../components/FloatingBtn';
import Button from '../components/Button';
import { withRouter } from 'react-router';
import { Icon } from 'react-icons-kit';
import {androidArrowBack} from 'react-icons-kit/ionicons/androidArrowBack'
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';


const New = ({history}) => {

    const [serverRes, setServerRes] = useState("UNSET");

    useEffect(() => {
        fetch("http://localhost:5001/gethome", {
                method: 'GET',
                "Access-Control-Allow-Origin": "*",
            }).then((response, err) => {
                response.json()
            }).catch(error => {
                if(error) setServerRes("NO_SERVER");
            })
    }, [])

    useEffect(() => {
        if(serverRes === "NO_SERVER"){
            toast.error('Can\'t Find Server. Checkout Learn More Section To Start It.', {
                position: "bottom-right",
                autoClose: 5000,
                hideProgressBar: false,
                closeOnClick: true,
                pauseOnHover: true,
                draggable: true,
                progress: undefined,
            });
        }
    }, [serverRes])

    return(
        <div className="new-main">
            <div className="col-1">
            <div id={"r"}>
                <Card history={history} task={"tabular"}/>
                <Card history={history} task={"text"}/>
            </div>
            <div id={"r"}>
                <Card history={history} task={"vision"}/>
                <Card history={history} task={"collab"}/>
            </div>
            </div>

            <div id={"learn-more"}>
            <a target="_blank" href={"https://github.com/manikyabard/DashAI/wiki"}>
                <div>
                    Learn More
                </div>
            </a>
            <a target="_blank" href={"https://fastai1.fast.ai/"}>
                <div>
                fast.ai
                </div>
            </a>
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


export default withRouter(New);
