import React from 'react';
import { Icon } from 'react-icons-kit';
import {play} from 'react-icons-kit/fa/play'
function FloatingBtn({showSearch, setShowSearch}) {
    return(
        <div 
        onClick={() => setShowSearch(!showSearch)}
        style={{
          display: showSearch ? "none":"flex"
          // backgroundColor: showSearch ? '#da2323':'',
          // transform: showSearch ? 'rotate(45deg)':'rotate(0deg)'
        }}
        className={'floating-btn'}>
          <Icon icon={play} size={30}/>
        </div>
    )
}

export default FloatingBtn;