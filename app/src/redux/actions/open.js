import { OPEN_PROJECT, 
    NEW_PROJECT, 
    DEFAULT_PAGE,
    COMPILE_PROJECT, LOGIN_PAGE} from "./type/open"

export function openProjectPage(val) {
    return {
        type: OPEN_PROJECT,
        payload: {
            open: val,
            whatPage: "open"
        }
    }
}

export function openNewProjectPage(val) {
    return {
        type: NEW_PROJECT,
        payload: {
            new: val,
            whatpage: "new"
        }
    }
}

export function openCompilePage(val) {
    return {
        type: COMPILE_PROJECT,
        payload: {
            compile: val,
            whatpage: "compile"
        }
    }
}

export function openLoginPage(val) {
    return {
        type: LOGIN_PAGE,
        payload: {
            login: val,
            whatpage: "login"
        } 
    }
}

export function defaultPage(val) {
    return {
        type: DEFAULT_PAGE,
        payload: {
            open: false,
            login: false,
            new: false,
            compile: false,
            whatpage: "default"
        }
    }
}