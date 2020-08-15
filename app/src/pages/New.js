import React from 'react';
import './New.css';
import Card from '../components/Card';
import ProjectConfig from '../components/ProjectConfig';
// import FloatingBtn from '../components/FloatingBtn';
import Button from '../components/Button';
import { withRouter } from 'react-router';



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
                <Card history={history} task={"Collaborative Filtering"}/>
            </div>
            </div>
        </div>
    )
}


export default withRouter(New);
