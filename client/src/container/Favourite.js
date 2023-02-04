import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, TextInput, TouchableOpacity, ScrollView } from 'react-native'
import { getData } from '../helper/helper';
import { useIsFocused } from '@react-navigation/native';
import { Feather, Entypo } from '@expo/vector-icons';
import { getSuggestions } from '../helper/helper';
import Item from '../component/favourite/Item';

const Favourite = ({ navigation }) => {
    const isFoucsed = useIsFocused()
    const [text, onChangeText] = React.useState('');
    const [search, setSearch] = useState([]);
    const [favourite, setFavourite] = useState([])

    const handleSearch = (userInput) => {

        let result = getSuggestions(favourite, userInput)
        if (userInput) {
            setSearch(result)
        }
        else {
            setSearch(favourite)
        }
    }

    useEffect(() => {
        if (isFoucsed) {
            getData("favourite", (data) => {
                if (data)
                 {
                    setFavourite(data)
                    setSearch(data)
                }
            })
        }

    }, [isFoucsed])

    useEffect(() => {
        handleSearch(text)
    }, [text])

    const handleNavigationToFavourite = (id, details) => {
        navigation.navigate('SingleFavourite', { id: id, details: details })
    }

    return (
        <View style={styles.container}>
            <View style={{ flexDirection: 'row', alignItems: 'center' }}>
                <View style={styles.textInputLayout}>
                    <Feather name="search" size={25} color="grey" />
                    <TextInput style={styles.textInput} placeholder="  Search By Sentence" onChangeText={onChangeText} value={text}
                    />

                </View>
                <View style={{ flexDirection: 'row', flexGrow: 6 }}>
                    <TouchableOpacity onPress={() => { onChangeText('') }} style={{ justifyContent: 'flex-end' }}>
                        <Entypo name="circle-with-cross" size={26} color="grey" />
                    </TouchableOpacity>
                </View>
            </View>

            <ScrollView style={{ margin: 20 }}>

                {
                    search.map((item, index) => {
                        return (
                            <Item details={item} handleNavigation={() => handleNavigationToFavourite(index, item)} />
                        )
                    })
                }


                {
                    search.length === 0 && <Text style={styles.text}>No Result Found</Text>
                }
            </ScrollView>
        </View>
    )
}

const styles = StyleSheet.create({
    container: {
        display: 'flex',
        flexGrow: 1,
        backgroundColor: 'white',

    },
    textInput: {
        fontSize: 18,

    },
    textInputLayout: {
        flexDirection: 'row',
        marginVertical: 10,
        marginHorizontal: 20,
        borderWidth: 1,
        borderRadius: 50,
        borderColor: 'grey',
        padding: 10,
        width: 300
    },
    text: {
        textAlign:'center',
        fontSize: 20,
        lineHeight: 21,
        fontWeight: 'bold',
        letterSpacing: 0.25,
        color: 'grey',
    },

})


export default Favourite
