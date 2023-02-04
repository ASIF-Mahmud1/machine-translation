import { URL } from "../config/config"

const translate = async (sentence) => {
    try {
        let response = await fetch(URL+'predictSentence', {
          method: 'POST',
          headers: {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({sentence})
        })
      return await response.json()
    } catch(err) {
      console.log(err)
    }
  }

export {
    translate
}


