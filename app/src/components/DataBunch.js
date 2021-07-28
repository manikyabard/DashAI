// import React from 'react';
// import JsonEditor from './Jsoneditor';
// import Button from './Button';
// import { connect } from 'react-redux';
// import { update_type, update_data } from '../redux/actions/data';

// const DataBunch = ({visibility, handlePop, task, data, setData}) => {
//     const handleVis = () => {
//         handlePop(false);
//     }

//     const handleChange = data => {
//         console.log("DataBunch", data);
//         setData(data);
//     }
//     return(
//         <div 
//         style={{
//             display: visibility ? "flex":"none"
//         }}
//         className={"pop-main"}>
//             <div className={"data-container"}>
//                 <JsonEditor onChange={handleChange} data={data[task]["input"]} Title={"input"}/>
//                 <div className={['btn-gp']}>
//                     <Button label={"Done"}/>
//                     <Button onClick={handleVis} label={"Cancel"}/>
//                 </div>
//             </div>
//         </div>
//     )
// }

// const stateToProps = (state) => {
//     return {
//         "task": state.payload.task,
//         "data": state.payload
//     }
// }

// const dispatchToProps = (Dispatch) => {
//     return {
//         "setData": (val) => Dispatch(update_data(val)),
//         "setType": (val) => Dispatch(update_type(val))
//     }
// }

// export default connect(stateToProps, dispatchToProps)(DataBunch);