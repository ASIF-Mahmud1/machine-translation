import React,{Fragment} from "react";
import { ToastAndroid ,Text,ActivityIndicator} from "react-native";

const showToast = (message,duration) => {
  duration ? ToastAndroid.show(message, ToastAndroid[duration]):
  ToastAndroid.show(message, ToastAndroid.LONG);
};

const Loader =({loading} )=>{
    return <Fragment>
        <ActivityIndicator size="large" color="#00ff00" animating={loading} />
    {
       loading && <Text style={{textAlign:'center'}}>Please Wait</Text>
    }
    </Fragment>
}


export {
  showToast, 
  Loader
}