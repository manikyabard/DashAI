import React from 'react';
const Button = ({history}) => {
    const handleClick = () => {
        history.push("/modelbuilder");
    }
    return(
        <div onClick={handleClick} className={"btn-main"}>
            <h3>
                Create
            </h3>
        </div>
    )
}

export default Button;