import data from '../../assets/data.json';
const { UPDATE_DATA, UPDATE_TYPE } = require("../actions/type/open");

const initState = {
    type: UPDATE_DATA,
    payload: data
}

export default function Data_Reducer(state = initState, method) {
    switch(method.type){
        case UPDATE_DATA:
            return {
                type: UPDATE_DATA,
                payload: method.payload.value
            }
        case UPDATE_TYPE:
            return {
                type: UPDATE_TYPE,
                payload: {
                    ...state.payload,
                    task: method.payload.value
                }
            }
        default: return state
    }
}