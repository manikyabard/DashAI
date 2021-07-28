import { OPEN_PROJECT, NEW_PROJECT, COMPILE_PROJECT, LOGIN_PAGE, DEFAULT_PAGE } from "../actions/type/open"

const initState = {
    type: DEFAULT_PAGE,
    payload: {
        open: false,
        new: false,
        compile: false,
        login: false,
        whatpage: "default"
    }
}

const openProjectReducer = (state = initState, action) => {
    switch (action.type) {
        case OPEN_PROJECT:
            return {
                ...state,
                type: OPEN_PROJECT,
                payload: {
                    ...state.payload,
                    open: action.payload.open,
                    whatpage: "open"
                }
            }
        case NEW_PROJECT:
            return {
                ...state,
                type: NEW_PROJECT,
                payload: {
                    ...state.payload,
                    new: action.payload.new,
                    whatpage: "new"
                }
            }
        case COMPILE_PROJECT:
            return {
                ...state,
                type: COMPILE_PROJECT,
                payload: {
                    ...state.payload,
                    compile: action.payload.compile,
                    whatpage: "compile"
                }
            }
        case LOGIN_PAGE:
            return {
                ...state,
                type: LOGIN_PAGE,
                payload: {
                    ...state.payload,
                    login: action.payload.login,
                    whatpage: "login"
                }
            }
        case DEFAULT_PAGE:
            return {
                ...state,
                type: DEFAULT_PAGE,
                payload: action.payload
            }

        default:
            return state
    }
}

export default openProjectReducer;