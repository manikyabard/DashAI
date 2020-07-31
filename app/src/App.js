import React from 'react';
import './App.css';
import { BrowserRouter as Router, Route, Switch } from "react-router-dom";

//import LoginPage from './pages/LoginPage'
//import Main from './pages/Main'
import { Provider } from "react-redux";
import store from "./redux/store";
import ModelMenu from './components/ModelMenu';
import ModelBuilder from './components/ModelBuilder';
import Footer from './components/Footer'
import New from './pages/New';
function App() {
  return (
    <Provider store={store}>
      <Router>
        {/*<Footer />*/}
        <Switch>
          <Route path="/modelbuilder">
          <div className={"app-main"}>
              <ModelMenu />

            </div>
          </Route>
          <Route path="/">
            <div className={"app-main"}>
              <New />
            </div>
          </Route>
        </Switch>
      </Router>
    </Provider>
  )
}

export default App;
