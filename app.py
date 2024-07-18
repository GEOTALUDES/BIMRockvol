import streamlit as st

#set the title
st.title ('Detección Automática del Volumen de Rocas')

#set header
st.header ('Please upload an image')

#Upload file
st.file_uploader ('',type=['png','jpg'.'jpeg'])

#Load the model
