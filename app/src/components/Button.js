import React from 'react';
const Button = ({label, onClick}) => {
    return(
        <div onClick={onClick} className={"btn-main"}>
            <h3>
                {label}
            </h3>
        </div>
    )
}

export default Button;