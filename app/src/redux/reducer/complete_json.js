const { UPDATE_DATA } = require("../actions/type/open");
import data from '../../assets/data.json';

const initState = {
    type: UPDATE_DATA,
    payload: data
}

export default function Data_Reducer(state = initState, method) {
    switch(method.type){
        case UPDATE_DATA:
            return {
                type: UPDATE_DATA,
                payload: {
                    ...state.payload,
                    data: method.value
                }
            }
    }
    
}