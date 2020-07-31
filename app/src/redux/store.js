import openProjectReducer from "./reducer/open";
import {createStore} from 'redux'


// const createStore = redux.createStore



const store = createStore(openProjectReducer)


export default store;