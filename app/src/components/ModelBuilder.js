import React, { useState, useRef } from 'react';
import DataBlock from './DataBlock';
import SearchPanel from './SearchPanel';


function ModelBuilder(props) {
    const SearchPanelRef =  useRef();
    const [showSearch, setShowSearch] = useState(false);
    return(
        <div ref={r => SearchPanelRef.current = r} 
            className={'model-builder'}>
            <DataBlock setShowSearch={setShowSearch} />
            <SearchPanel  
            setShowSearch={setShowSearch}
            showSearch={showSearch} />
        </div>
    )
}

export default ModelBuilder;