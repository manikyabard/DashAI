import React from 'react';
import JsonFiled from './JsonFiled';
import Button from './Button';

const PopPage = ({visibility, onClick}) => {

    const handlePop = () => {
        onClick(false);
    }
    return(
        <div 
        style={{
            display: visibility ? "flex":"none"
        }}
        className={"pop-main"}>
            <div className={"container"}>
                <JsonFiled label={"nn."} placeholder={"Any PyTorch Supported Layer"}/>
                <div className={'btn-gp'}>
                    <Button label={"Add"}/>
                    <Button onClick={handlePop} label={"Cancel"}/>
                </div>
            </div>
        </div>
    )
}


export default PopPage;