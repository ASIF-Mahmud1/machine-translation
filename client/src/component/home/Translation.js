import React, {useState} from 'react';
import {Text,View, StyleSheet,TouchableOpacity, TextInput, Clipboard} from 'react-native'
import {  Ionicons ,AntDesign ,MaterialIcons} from '@expo/vector-icons';
import { showToast } from '../../helper/component/Indicator.js';
const Translation=({language,translate, handleAddToFavourite})=>{
    const handleCopyToClipboard=()=>{
        if(translate)
        {
            Clipboard.setString(translate)
           
            showToast("Copied to Clipboard")
        }
     
    }
    const handleShare=()=>{
        showToast("This feature is in progress")
    }

    const handleFavourite=()=>{
        handleAddToFavourite()
        showToast("Added to Favourite !")
    }
    return (
        <View style={styles.container}>
            <View style={styles.top} >
               <View style={styles.topLeft}>
                   <Text style={{fontSize:20,fontWeight:'bold',  color:'orange'}}>{language}</Text>  
               </View> 
        
            </View>
            <TextInput value={translate} placeholder='Translate to English'  multiline={true}  style={styles.input} selectTextOnFocus={false}  editable={false} />
            <View  style={styles.topRight}>
                   <TouchableOpacity onPress={handleCopyToClipboard} >
                      <Ionicons name="copy" size={24} color="black" />
                   </TouchableOpacity>
                   <TouchableOpacity  onPress={handleShare}>
                       <AntDesign name="sharealt" size={24} color="black" />
                   </TouchableOpacity>
                   <TouchableOpacity  onPress={handleFavourite}>
                       <MaterialIcons name="favorite-outline" size={24} color="black" />
                       {/* <MaterialIcons name="favorite" size={24} color="black" /> */}
                   </TouchableOpacity>
                   
               </View>
        </View>
     
    )
}


const styles = StyleSheet.create({ 
     container:{
      display:'flex',
      flexGrow:1,   
      backgroundColor:'white',
      paddingVertical:30
     },
     top:{
       flexDirection:'row',
       justifyContent:'space-between'
     },
     topLeft:{
        flexGrow:3,
        justifyContent:'center'
     },
     topRight:{
         flexDirection:'row',
         marginVertical:10,
         paddingVertical:10,
       //  borderWidth:1,
         justifyContent:'space-around',
         alignItems:'center'
     },
     input:{
        borderColor:'#ffa600',
        borderWidth:2,
        borderRadius:20,
        marginTop:15,
        paddingHorizontal:15,
       // paddingVertical:60,
      //  color:'white',
        height:100,
        fontSize:20,
        color:'black',
     }

})

export default Translation
