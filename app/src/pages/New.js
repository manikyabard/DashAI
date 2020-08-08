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
                <Card task={"Tabular"}/>
                <Card task={"Text"}/>
            </div>
            <div id={"r"}>
                <Card task={"Vision"}/>
                <Card task={"Collaborative Filtering"}/>
            </div>
            <Button history={history}/>
            </div>
            <div className="col-2">
                <ProjectConfig />
            </div>
        </div>
    )
}


export default withRouter(New);
