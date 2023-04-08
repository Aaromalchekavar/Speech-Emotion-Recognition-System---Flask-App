import librosa
import numpy as np
import tensorflow as tf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
import pandas as pd

filepath = './Audio_features_All_pr.csv'

# Load the saved model
model = tf.keras.models.load_model('./emotion-recognition.hdf5')

def additional_preprocess(filepath):
  #read the csv file of extrated features
  df = pd.read_csv(filepath)
  df["Label"] = df["Label"].str.replace("sadness", "sad", case = True)
  df["Label"] = df["Label"].str.replace("happiness", "happy", case = True)
  df["Label"] = df["Label"].str.replace("Sad", "sad", case = True)
  df["Label"] = df["Label"].str.replace("anger", "angry", case = True)
  return df
  
#this fucntion is used to get audio features perform one hot encoding
def audio_features_final():
  df = additional_preprocess("./Audio_features_All_pr.csv")
  #get all the aduio features as numpy array from the dataframe 
  #last column is label so last column is not fetched only 0to:-1
  data=df[df.columns[0:-1]].values
  #perform one hot encoding on labels
  encoder = OneHotEncoder()
  #fetch the last column of labels and perform one hot encoding on them
  label=df["Label"].values
  label = encoder.fit_transform(np.array(label).reshape(-1,1)).toarray()
  #min max scaler is used to normalize the data
  scaler = MinMaxScaler()
  data=scaler.fit_transform(data)
  return encoder

#encoder for realtime recording
encoder = audio_features_final()

def noise(data):
  noise_amp = 0.035*np.random.uniform()*np.amax(data)
  data = data + noise_amp*np.random.normal(size=data.shape[0])
  return data

#fuction to strech audio
def stretch(data, rate=0.8):
  return librosa.effects.time_stretch(y=data,rate=0.8) 

#fucntion to shift audio range
def shift(data):
  shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
  return np.roll(data, shift_range)

#function to change pitch
def pitch(data, sampling_rate, pitch_factor=0.7):
  return librosa.effects.pitch_shift(y=data, sr=sampling_rate, n_steps=pitch_factor)

def extract_features(data,sample_rate):  
  
  #zero crossing rate
  result = np.array([])
  zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
  result = np.hstack((result, zcr)) 
  #print('zcr',result.shape)

  #chroma shift
  stft = np.abs(librosa.stft(data))
  chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
  result = np.hstack((result, chroma_stft))
  #print('chroma',result.shape)
  
  #mfcc
  mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
  result = np.hstack((result, mfcc))
  #print('mfcc',result.shape)
  
  #rmse
  rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
  result = np.hstack((result, rms)) 
  #print('rmse',result.shape)
  
  #melspectogram
  mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
  result = np.hstack((result, mel)) 
  #print('mel',result.shape)    

  #rollof 
  rollof = np.mean(librosa.feature.spectral_rolloff(y=data, sr=sample_rate).T, axis=0)
  result = np.hstack((result, rollof))
  #print('rollof',result.shape) 

  #centroids 
  centroid = np.mean(librosa.feature.spectral_centroid(y=data, sr=sample_rate).T, axis=0)
  result = np.hstack((result, centroid))
  #print('centroids',result.shape)

  #contrast
  contrast = np.mean(librosa.feature.spectral_contrast(y=data, sr=sample_rate).T, axis=0)
  result = np.hstack((result, contrast))
  #print('contrast',result.shape)

  #bandwidth
  bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=data, sr=sample_rate).T, axis=0)
  result = np.hstack((result, bandwidth))
  #print('bandwidth',result.shape)

  #tonnetz
  tonnetz = np.mean(librosa.feature.tonnetz(y=data, sr=sample_rate).T, axis=0)
  result = np.hstack((result, tonnetz))
  #print('tonnetz',result.shape) 

  return result
  
def get_features_recorded(data,sr): 

  #get features for recorded audio using mircophone
  res1 = extract_features(data,sr)
  result = np.array(res1)
  
  #get audio features with noise
  noise_data = noise(data)
  res2 = extract_features(noise_data,sr)
  result = np.vstack((result, res2))
    
  #get audio features with stretching and pitching
  new_data = stretch(data)
  data_stretch_pitch = pitch(new_data, sr)
  res3 = extract_features(data_stretch_pitch,sr)
  result = np.vstack((result, res3))
  
  return result

def realtimepredict(file_path):

    # Load the audio file and compute the mel spectrogram
    audio, sr = librosa.load(file_path, sr=16000)
    res_model = load_model("./emotion-recognition.hdf5")

    audio = audio.astype('float')
    #get audio features from the recorded voice
    feature = get_features_recorded(audio,sr)
    #apply min max scaling
    scaler = MinMaxScaler()
    feature = scaler.fit_transform(feature)
    #get the predicted label
    label=res_model.predict(feature)
    #get the label information by reversing one hot encoded output
    label_predicted=encoder.inverse_transform(label)
    return label_predicted[0][0]
