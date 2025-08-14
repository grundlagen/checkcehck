import React from 'react';
import { SafeAreaView, View, Text, TextInput, Button, StyleSheet } from 'react-native';

/**
 * QuackCoin mobile app skeleton.
 *
 * This component provides a minimal UI for the QuackCoin proof‑of‑concept.  It
 * does not implement audio recording or wallet integration yet.  Use it as
 * a starting point for building a richer React Native or Expo application.
 */
export default function App() {
  const [stake, setStake] = React.useState('');

  const handleSubmit = () => {
    // Placeholder submit handler.  In a real app you would record audio,
    // upload the file to IPFS via the backend and then call the smart
    // contracts to stake and hatch the egg.
    alert(`Quack submitted with stake of ${stake} QC (not yet implemented)`);
  };

  return (
    <SafeAreaView style={styles.container}>
      <Text style={styles.title}>QuackCoin</Text>
      <Text style={styles.subtitle}>Prototype Mobile Client</Text>
      <View style={styles.form}>
        <TextInput
          style={styles.input}
          placeholder="Stake (QC)"
          keyboardType="numeric"
          value={stake}
          onChangeText={setStake}
        />
        <Button title="Record & Submit" onPress={handleSubmit} />
      </View>
      <Text style={styles.note}>
        This is a barebones UI.  Recording, staking and playback are not yet
        implemented.
      </Text>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#f0f9ff',
    padding: 16,
  },
  title: {
    fontSize: 32,
    fontWeight: 'bold',
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 16,
    marginBottom: 24,
  },
  form: {
    width: '100%',
    paddingHorizontal: 24,
  },
  input: {
    borderWidth: 1,
    borderColor: '#ccc',
    borderRadius: 4,
    padding: 8,
    marginBottom: 12,
  },
  note: {
    marginTop: 24,
    color: '#666',
    textAlign: 'center',
  },
});