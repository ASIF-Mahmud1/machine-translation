import React from 'react';
import { StyleSheet, Text, View, TouchableOpacity, Clipboard } from 'react-native';

import { AntDesign, Ionicons } from '@expo/vector-icons';
import { showToast } from '../../helper/component/Indicator';
import { deleteFavourite } from '../../helper/helper';
export default function SingleFavourite({ route, navigation }) {
    const { orginalLanguage, translateLanguage, orginalSentence, translateSentence, id } = route.params.details;

    const handleCopyToClipboard=(sentence)=>{
     
            Clipboard.setString(sentence)
           
            showToast("Copied to Clipboard")
    }

    const handleDelete=()=>{
        deleteFavourite(id,(data)=>{
            if(data)
            {
                showToast("Deleted Succesfully")
                navigation.goBack()
            }
        })
    }

    return (
        <View style={{ display: 'flex', flexGrow: 1, backgroundColor: 'white' }} >
            <View style={styles.container}>
                <Text style={styles.orginalText}>      {orginalLanguage} </Text>

                <TouchableOpacity style={styles.btn} disabled={true}>

                    <View style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' }}>
                        <View>
                            <Text style={styles.orginal}>{orginalSentence}</Text>
                       
                        </View>

                        <View>
                        <TouchableOpacity onPress={()=> handleCopyToClipboard(orginalSentence)} >
                               <Ionicons name="copy" size={24} color="#71DEA3" />
                          </TouchableOpacity> 
                     
                        </View>

                    </View>

                </TouchableOpacity>

                <Text style={styles.translateText}>     {translateLanguage} </Text>

                <TouchableOpacity style={styles.btn} disabled={true}>

                    <View style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' }}>
                        <View>
                           
                            <Text style={styles.translate}>{translateSentence} </Text>
                        </View>

                        <View>
                           <TouchableOpacity  onPress={()=> handleCopyToClipboard(translateSentence)} >
                               <Ionicons name="copy" size={24} color="#71DEA3" />
                          </TouchableOpacity> 
                     

                        </View>

                    </View>

                </TouchableOpacity>
                <TouchableOpacity onPress={handleDelete} style={{marginTop:20,backgroundColor:'red', borderRadius:8,  height:40,width:150, alignSelf:'center', alignItems:'center', justifyContent:'center'}} >
                    <Text  style={{color:'white', fontWeight:"bold", fontSize:16}} >Delete</Text>
                </TouchableOpacity>
            </View>


        </View>
    );
}

const styles = StyleSheet.create({
    container: {
        display: 'flex',
        marginBottom: 15,
        borderRadius: 10,
        marginHorizontal: 15,
        //    alignItems:'center',
        //  justifyContent:'center',
        flexGrow: 1
    },

    btn: {
        width: 350,
        backgroundColor: '#fff',
        paddingVertical: 15,
        paddingHorizontal: 20,
        borderWidth: 2,
        marginHorizontal: 15,
        marginVertical: 10,
        borderRadius: 10,
        borderColor: '#71DEA3'
    },
    orginal:
    {
        color: 'crimson',
        fontSize: 18,
        fontWeight: 'bold',
        fontStyle: "italic",
        marginRight: 5
    },
    translate: {
        color: 'orange',
        fontSize: 18,
        fontWeight: 'bold',
        fontStyle: "italic",
        marginRight: 5
    },
    orginalText: {
        color: 'crimson',
        fontSize: 12,
        fontWeight: 'bold',
        fontStyle: "italic",
        marginRight: 5
    },
    translateText:{
        color: 'orange',
        fontSize: 12,
        fontWeight: 'bold',
        fontStyle: "italic",
        marginRight: 5
    },
});
