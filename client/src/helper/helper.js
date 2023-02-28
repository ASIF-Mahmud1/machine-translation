import { AsyncStorage} from 'react-native'

const getData = async (key, callback) => {
    try {
        const value = await AsyncStorage.getItem(key)
        if (value != null) {
            callback(JSON.parse(value))
        }
        else {
            callback(null)
        }
    } catch (e) {
        callback(null)
        console.log('Unable to get Data', e)
    }
}

const storeData = async (key, value, callback) => {
    try {
        await AsyncStorage.setItem(key, JSON.stringify(value))
        callback && callback()
    } catch (e) {
        console.log('Unable to store Data', e)
    }
}

const deleteFavourite = async (favouriteId, callback) => {
    try {
        const value = await AsyncStorage.getItem("favourite")
        if (value != null) {
            const result= (JSON.parse(value)).filter((item)=> item.id !== favouriteId  )
            storeData("favourite",result)
            callback(result)
        }
        else {
            callback(null)
        }
    } catch (e) {
        callback(null)
        console.log('Somethin went wrong', e)
    }
}

const truncate=(string, limit)=>{
    if (string.length > limit) {
    return string.substring(0, limit-1) + "...";
    }
    return string
  }

  const getSuggestions=(list,userInput)=> {
 
    let formattedUserInput= userInput.toLowerCase().trim()
    let userInputLength =formattedUserInput.length
  
    return list.filter((translation) => {
  
      if ((translation.orginalSentence.substr(0, userInputLength).toLowerCase().trim() === formattedUserInput) || (translation.translateSentence.substr(0, userInputLength).toLowerCase().trim() === formattedUserInput)  ) {
        return true
      } else {
        return false
      }
    })
  
    }



export { 

    getData, 
    storeData,
    truncate,
    getSuggestions,
    deleteFavourite
}