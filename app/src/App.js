import React, { useState, useEffect } from 'react';
import './App.css';
import { BrowserRouter as Router, Route, Switch } from "react-router-dom";

//import LoginPage from './pages/LoginPage'
//import Main from './pages/Main'
import { Provider, connect } from "react-redux";
import store from "./redux/store";
import ModelMenu from './components/InputMenu';
import ModelBuilder from './components/ModelBuilder';
// import Footer from './components/Footer'
import New from './pages/New';
import TunnelPage from './components/TunnelPage';
import FloatingBtn from './components/FloatingBtn';
// import FloatingBtn from './components/FloatingBtn';
import 'bootstrap/dist/css/bootstrap.min.css';
import CoreMenu from './components/CoreMenu';
import ModelGen from './components/ModelMenu';
import TrainMenu from './components/TrainMenu';
import VerumMenu from './components/VerumMenu';

function App({data}) {
  const [training, setTraining] = useState(false);

  return (
    
      <Router>
        {/*<Footer />*/}
        <Switch>
          <Route path="/task">
              <div className={"app-main"}>
                <ModelMenu />
                {/*<ModelBuilder />*/}
              </div>
          </Route>

          <Route path="/core">
              <div className={"app-main"}>
                <CoreMenu />
              </div>
          </Route>

          <Route path="/model">
              <div className={"app-main"}>
                <ModelGen />
              </div>
          </Route>

          <Route path="/train">
              <div className={"app-main"}>
                <TrainMenu />
                <TunnelPage visibility={training} setVisibility={setTraining}/>
                <FloatingBtn showSearch={training} setShowSearch={setTraining}/>
              </div>
          </Route>

          <Route path="/verum">
              <div className={"app-main"}>
                <VerumMenu />
                <TunnelPage visibility={training} setVisibility={setTraining}/>
                <FloatingBtn showSearch={training} setShowSearch={setTraining}/>
              </div>
          </Route>

          <Route path="/">
          <div className={"app-main"}>
              <New />
            </div>
          </Route>
        </Switch>
      </Router>
  )
}


const stateToProps = (state) => {
  return {
      "data": state.payload
  }
}

const dispatchToProps = (Dispatch) => {
  return {
      
  }
}


export default connect(stateToProps, dispatchToProps)(App);
