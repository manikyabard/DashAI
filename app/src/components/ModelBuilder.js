import React, { useState, useRef } from 'react';
import DataBlock from './DataBlock';
import SearchPanel from './SearchPanel';
import PopPage from './PopPage';
import DataBunch from './DataBunch';
import FloatingBtn from './FloatingBtn';
import TunnelPage from './TunnelPage';


function ModelBuilder(props) {
    const SearchPanelRef =  useRef();
    const [showSearch, setShowSearch] = useState(false);
    const [data, setData] = useState(false);
    const [training, setTraining] = useState(false);
    return(
        <div ref={r => SearchPanelRef.current = r} 
            className={'model-builder'}>
            <DataBunch visibility={data} handlePop={setData}/>
            <DataBlock setShowSearch={setData} setAddlayer={setShowSearch} />
            <PopPage visibility={showSearch} onClick={setShowSearch}/>
            <TunnelPage visibility={training} setVisibility={setTraining}/>
            <FloatingBtn showSearch={training} setShowSearch={setTraining}/>
            {/*<SearchPanel  
            setShowSearch={setShowSearch}
            showSearch={showSearch} />*/}
        </div>
    )
}

export default ModelBuilder;