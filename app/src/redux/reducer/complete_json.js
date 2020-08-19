import data from '../../assets/data.json';
import verum from '../../assets/verum.json';
import train from '../../assets/train.json';
const { UPDATE_DATA, UPDATE_TYPE } = require("../actions/type/open");

const initState = {
    type: UPDATE_DATA,
    payload: {
        "data":data,
        "verum": verum,
        "train": train
    }
}

export default function Data_Reducer(state = initState, method) {
    switch(method.type){
        case UPDATE_DATA:
            return {
                type: UPDATE_DATA,
                payload: {
                    ...state.payload,
                    [method.level]: method.payload.value
                }
            }
        case UPDATE_TYPE:
            return {
                type: UPDATE_TYPE,
                payload: {
                    ...state.payload,
                    "data": {
                        ...state.payload.data,
                        task: method.payload.value
                    }
                }
            }
        default: return state
    }
}