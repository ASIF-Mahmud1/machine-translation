import React, { useState, useEffect } from 'react';
import {Text,View, StyleSheet} from 'react-native'
import { FontAwesome ,Entypo,AntDesign } from '@expo/vector-icons';
import ConverterCell from '../component/home/ConverterCell';
import Sentence from '../component/home/Sentence';
import Translation from '../component/home/Translation';
import { translate } from '../api/translate-api';
import { Loader,showToast } from '../helper/component/Indicator';
import {getData, storeData } from '../helper/helper';
import ModalPage from '../helper/component/Modal';

const Home=()=>{
    const [language, setLanguage]= useState({
        orginal:'French',
        translate:'English'
    })

    const [sentence, setSentence]= useState({
        orginal:'',
        translate:''
    })

    const[loading,setLoading]=useState(false)
    const [selectLanguage,setSelectLanguage] = useState({
        type: ''  , // orignal or translate
        language: ''  // english, french ...
     })
    const[modalVisible, setModalVisible] = useState(false)
 
  useEffect(()=>{
 
   if(selectLanguage.language!=="" && selectLanguage.type !== "")
   {
     let key= selectLanguage.type
     let value =selectLanguage.language  
    setLanguage({...language, [key]: value })
   }
   
  },[selectLanguage])

    const handleTranslate= async()=>{
        setLoading(true)
        const result= await translate(sentence.orginal)
        if(result?.prediction)
        {
           
            setSentence({...sentence, translate: result.prediction})
        }
        else 
        {
           showToast("Something went wrong")
        }
        setLoading(false)
     
    }

    const handleAddToFavourite =()=>{

        if( sentence.orginal &&  sentence.translate )
        {

            const translationDetails =
            {
                orginalLanguage: language.orginal,
                translateLanguage: language.translate,
                orginalSentence: sentence.orginal,
                translateSentence: sentence.translate,
                id: Date.now() +Math.random()
            }

            getData('favourite', (data) => {
                let result = []

                if (data)   // already exist in Async
                {
                    result = [...data]
                    result.push(translationDetails)

                }
                else   // create new array of favourite
                {
                    result.push(translationDetails)
                }

                storeData('favourite', result)
            })
        }

        else 
        {
            showToast("Cannot save to Favourite")
        }

    }

    return (
        <View style={styles.container}>
              <Loader loading={loading}/>
              <ConverterCell orginal={language.orginal}  translate={language.translate}  handleParentState= {(sentenceType)=>{ setSelectLanguage({language: language[sentenceType],type:sentenceType});  setModalVisible(true) }}   />     
              <Sentence language={language.orginal} sentence={sentence.orginal}  handleParentState= {(feedBack)=> setSentence({...sentence, orginal:feedBack}) } handleTranslate={handleTranslate} />
              <Translation language={language.translate} translate= {sentence.translate} handleAddToFavourite ={handleAddToFavourite}  />
              <ModalPage modalVisible={modalVisible} setModalVisible={setModalVisible} title={"Select Language"} status={selectLanguage.language} handleParentState={( value)=>{setSelectLanguage({...selectLanguage,language:value}) }}  />
        </View>
     
    )
}


const styles = StyleSheet.create({ 
     container:{
      display:'flex',
      flexGrow:1,   
      backgroundColor:'white',
      paddingHorizontal:20,
     
     },

})

export default Home

/*

orginalLanguage
translateLanguage

orginalSentence
translateSentence
*/