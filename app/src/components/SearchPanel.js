import React, { useState, useRef, useEffect } from 'react';
import { Icon } from 'react-icons-kit';
import {iosSearchStrong} from 'react-icons-kit/ionicons/iosSearchStrong'
import {androidAddCircle} from 'react-icons-kit/ionicons/androidAddCircle'


function SearchPanel({showSearch, setShowSearch}) {
  const [search, setSearch] = useState('');
  const inputRef = useRef();
  const handleSearch = () => {
    setShowSearch(false);
  }

  const handleChange = e => {
    setSearch(e.target.value);
  } 

  useEffect(() => {
    if(showSearch === true){
      inputRef.current.focus();
    }
    else {
      inputRef.current.blur();
      setSearch('');
    }
  }, [showSearch])

  return (
      <div 
      style={{
        height: (search !== '' && showSearch !== false) ? '120px':'',
        transform: showSearch ? 'translateY(130px)':'translateY(0)'
      }}
      className={'search-panel'}>
      <div className={'search-container'}>
        <div className={'search-icon'}>
        <Icon icon={iosSearchStrong} size={30} />
        </div>
        <div className={'search-input'}>
          <input ref={inputRef} value={search} 
          onChange={e => handleChange(e)} placeholder={'Any PyTorch Supported layer'} />
        </div>
        <div className={'cancel'}>
          <Icon icon={androidAddCircle} 
          onClick={handleSearch}
          size={25}
          style={{transform: 'rotate(45deg)'}} />
        </div>
      </div>
        
      </div>
  )
}

export default SearchPanel;