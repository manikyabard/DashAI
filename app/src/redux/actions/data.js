import { UPDATE_DATA, UPDATE_TYPE } from './type/open';


export const update_data = (val) => {
    return {
        type: UPDATE_DATA,
        payload: {
            value: val,
        }
    }
}

export const update_type = val => {
    return {
        type: UPDATE_TYPE,
        payload: {
            value: val,
        }
    }
}