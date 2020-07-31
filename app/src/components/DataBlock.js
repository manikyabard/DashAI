import React from 'react';
import { Icon } from 'react-icons-kit';
import {androidAdd} from 'react-icons-kit/ionicons/androidAdd'
import {database} from 'react-icons-kit/fa/database'
function DataBlock({setShowSearch}) {

    const handleSearch = () => {
        setShowSearch(true);
    }

    return(
        <div className={'data-block'}>
            <div onClick={handleSearch} className={'connector'}>
                <Icon icon={androidAdd} size={30} />
            </div>
            <div className={'db-col1'}>
                <Icon icon={database} size={35} />
            </div>
            <div className={'db-col2'}>
                <input disabled={true} placeholder={'Data Source'}/>
            </div>
        </div>
    )
}

export default DataBlock;