import { UPDATE_DATA, UPDATE_TYPE } from './type/open';


export const update_data = (val, level = "data") => {
    return {
        type: UPDATE_DATA,
        level: level,
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

