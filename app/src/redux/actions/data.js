import { UPDATE_DATA, UPDATE_PROJECT } from './type/open';


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
        type: UPDATE_PROJECT,
        payload: {
            value: val,
        }
    }
}