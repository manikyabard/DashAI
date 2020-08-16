import React, { useEffect } from 'react';
import Button from './Button';
import { connect } from 'react-redux';


const TunnelPage = ({visibility, setVisibility, data}) => {
    const handlePop = () => {
        setVisibility(false);
    }

    useEffect(() => {
        fetch("http://localhost:5000/generate/", {
            method: 'POST',
            body: JSON.stringify(data)
        })
    }, [])

    return(
        <div 
        style={{
            display: visibility ? "flex":"none"
        }}
        className={"pop-main"}>
            <div className={"training-page"}>
                <div className={"console"}>
                    <p>console view</p>
                </div>
                <div className={"header"}>
                    <div className={'btn-gp'}>
                        <Button onClick={handlePop} label={"Cancel"} type={"close"}/>
                    </div>
                </div>
            </div>
        </div>
    )
}


const stateToProps = (state) => {
    return {
        "data": state.payload,
    }
}

const dispatchToProps = (Dispatch) => {
    return {
        
    }
}


export default connect(stateToProps, dispatchToProps)(TunnelPage);