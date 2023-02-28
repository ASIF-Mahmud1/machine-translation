import * as React from 'react';
import { createDrawerNavigator } from '@react-navigation/drawer';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import Home from '../container/Home';
import Favourite from '../container/Favourite';
import SingleFavourite from '../component/favourite/SingleFavourite';



////////////////////////////////////////////////////////////////////////////////////

const Stack = createNativeStackNavigator();

function FavouriteStack() {
  return ( <Stack.Navigator>
        <Stack.Screen name="Favourite" component={Favourite}   options={{headerShown:false}}/>
        <Stack.Screen name="SingleFavourite" component={SingleFavourite}   options={{headerShown:false}} />
      </Stack.Navigator>
  );
}

//////////////////////////////////////////////////////////////////////////////////

const Drawer = createDrawerNavigator();

export default function App() {
  return (
    <NavigationContainer>
      <Drawer.Navigator initialRouteName="Home">
        <Drawer.Screen name="Home" component={Home} />
        <Drawer.Screen name="Favourites" component={FavouriteStack} />
      </Drawer.Navigator>
    </NavigationContainer>
  );
}