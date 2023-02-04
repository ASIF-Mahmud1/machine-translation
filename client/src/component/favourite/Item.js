import React from 'react';
import { StyleSheet, Text, View, TouchableOpacity ,Image} from 'react-native';

import { AntDesign } from '@expo/vector-icons';
import { truncate } from '../../helper/helper';
export default function Item({details, handleNavigation}) {
  const { orginalLanguage,translateLanguage,orginalSentence,translateSentence} =details
 

  
  return (
    <View style={styles.container}>


       <TouchableOpacity style={styles.btn} onPress={ handleNavigation  } >
       
       
        <View style={{flexDirection:'row',justifyContent: 'space-between',alignItems:'center'}}>
           <View>
           <Text style={styles.orginal}>{truncate(orginalSentence,20)}</Text>
        <Text style={styles.translate}>{truncate(translateSentence,20)} </Text>
        <Text style={styles.category}>{orginalLanguage} to  {translateLanguage} </Text>
           </View>   

           <View>
             <AntDesign name="rightcircleo" size={24} color="black" />
          </View>   

        </View>
      
      </TouchableOpacity>
        
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    display:'flex',
    marginBottom:15,
    borderRadius:10,
    marginHorizontal:15,
    alignItems:'center',
    justifyContent:'center'
  },

  btn:{
      width:350,
        backgroundColor: '#fff',
        paddingVertical: 15,
        paddingHorizontal: 20,
        borderWidth: 2,
        marginHorizontal: 15,
        marginVertical: 10,
        borderRadius: 10,
        borderColor: '#71DEA3'
    },
  orginal :
  {
    color:'crimson',
   fontSize:18,
   fontWeight:'bold',
   fontStyle:"italic",
   marginRight:5
  },
  translate:{
    color:'orange',
   fontSize:18,
   fontWeight:'bold',
   fontStyle:"italic",
   marginRight:5
  },
  category:{
    color:'#2F4F4F',
    fontSize:12,
    fontWeight:'bold',
    fontStyle:"italic",
    marginRight:5
  }
});
