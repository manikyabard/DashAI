import React from 'react';
import './New.css';
import Card from '../components/Card';
import ProjectConfig from '../components/ProjectConfig';
// import FloatingBtn from '../components/FloatingBtn';
import Button from '../components/Button';
import { withRouter } from 'react-router';
import { Icon } from 'react-icons-kit';
import {androidArrowBack} from 'react-icons-kit/ionicons/androidArrowBack'


const New = ({history}) => {
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
            <a target="_blank" href={"https://github.com/manikyabard/DashAI"}>
                <div>
                    Learn More
                </div>
            </a>
            <a target="_blank" href={"https://github.com/fastai/fastai1"}>
                <div>
                fast.ai
                </div>
            </a>
            </div>
        </div>
    )
}


export default withRouter(New);
