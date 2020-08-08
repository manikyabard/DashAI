import React from 'react';
import JsonEditor from './Jsoneditor';
import Button from './Button';

const data = {
    "csv_name": "./data/hello.csv",
    "dep_var": "columnt",
    "cat_names": [
        "column1"
    ],
    "cont_names": [
        "column2"
    ],
    "test_df": {
        "has_test": false,
        "csv_name": null
    }
}

const DataBunch = ({visibility, handlePop}) => {
    const handleVis = () => {
        handlePop(false);
    }
    return(
        <div 
        style={{
            display: visibility ? "flex":"none"
        }}
        className={"pop-main"}>
            <div className={"data-container"}>
                <JsonEditor data={data} Title={"Data"}/>
                <div className={['btn-gp']}>
                    <Button label={"Done"}/>
                    <Button onClick={handleVis} label={"Cancel"}/>
                </div>
            </div>
        </div>
    )
}

export default DataBunch;