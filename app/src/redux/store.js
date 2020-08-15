import {createStore} from 'redux'
import Data_Reducer from "./reducer/complete_json";


// const createStore = redux.createStore



const store = createStore(Data_Reducer)


export default store;