import React from 'react';
import Button from './Button';


const TunnelPage = ({visibility, setVisibility}) => {
    const handlePop = () => {
        setVisibility(false);
    }
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

export default TunnelPage;