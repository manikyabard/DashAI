import React from 'react';
import { Icon } from 'react-icons-kit';
import {androidAdd} from 'react-icons-kit/ionicons/androidAdd'
function FloatingBtn({showSearch, setShowSearch}) {
    return(
        <div 
        onClick={() => setShowSearch(!showSearch)}
        style={{
          backgroundColor: showSearch ? '#da2323':'',
          transform: showSearch ? 'rotate(45deg)':'rotate(0deg)'
        }}
        className={'floating-btn'}>
          <Icon icon={androidAdd} size={50}/>
        </div>
    )
}

export default FloatingBtn;