import React, { useState, useEffect, Fragment } from 'react';
import { Platform, Alert, Modal, StyleSheet, Text, Pressable, View, TouchableWithoutFeedback, Keyboard, TextInput, Dimensions, KeyboardAvoidingView, TouchableOpacity } from 'react-native';
import { FontAwesome ,Entypo } from '@expo/vector-icons';



const StatusList =({currentStatus,handleParentState })=>{
 // alert(currentStatus)
  const statusList= ["Bengali","English","French", "Hindi", ]
  return (
    <Fragment>
    {
      statusList.map((item)=>{
           return  <TouchableOpacity key= {item} style={{flexDirection:'row', alignItems:'center',paddingVertical:10}} onPress={()=>{handleParentState(item)}}>
           <Text style={{width:200, fontSize:20,fontStyle:'italic'}}>{item}</Text>
            {
              item.toLowerCase()===currentStatus.toLowerCase() &&  <Entypo style={{ marginLeft: 10,color:'green' }} name="check" size={24} color="black" />
            }
          
         </TouchableOpacity>
      })
    }
    </Fragment>
  )
}
const ModalPage = ({ modalVisible, setModalVisible,title,status,handleParentState })=> {
    const [taskStatus,setTaskStatus]=useState(status)
    useEffect(()=>{
      setTaskStatus(status)
    
     
    },[status])
     useEffect(()=>{
        handleParentState(taskStatus)
   
     },[taskStatus])

    return(
        <View>
          
        <Modal
            animationType="slide"
            transparent={false}
            visible={modalVisible}
            onRequestClose={() => {
                setModalVisible(!modalVisible)
            }}>
          <View style={styles.container}>
          <Text>Lan is {status}</Text>
              <Text style={styles.title}>{title}</Text>
              <View>
                <StatusList currentStatus= {taskStatus} handleParentState={(value)=>{setTaskStatus(value)}} />
              </View>
              <TouchableOpacity style={styles.confirm} onPress={()=>{ setModalVisible(!modalVisible) ;  }}>
                <Text style={styles.confirmText}>Confirm</Text>
              </TouchableOpacity>
          </View>
         
        </Modal>
    </View>
    )
}

const styles = StyleSheet.create({
    container: {
      flex: 1,
      flexDirection: 'column',
      alignItems:'center',
      paddingHorizontal: 40,
      paddingTop: 40
    },
    title:{
      fontSize:25,
      fontStyle:'italic',
      color:'grey'
    },
    confirm:{
      height:50,
      flexDirection: 'row',
      justifyContent: 'center',
      alignItems: 'center',
      backgroundColor:'#301934',
      width: 300,
      marginBottom:20,
      borderRadius:10,
    },
    confirmText:{
      color:"white", 
      fontWeight:"bold",
      fontSize:20
    }

});


export default ModalPage;