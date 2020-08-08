import React from 'react';
import './Card.css';

const Card = ({task}) => {
    return(
        <div className="main">
            <h3>{task}</h3>
        </div>
    )
}


export default Card;