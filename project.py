import streamlit as st 
import pickle 
import numpy as np
pipe=pickle.load(open('pipe.pkl', 'rb'))
df=pickle.load(open('df.pkl', 'rb'))
st.title("Laptop Price Predictor")
company=st.selectbox("Brand",df['Company'].unique())

type=st.selectbox("Type",df['TypeName'].unique())

ram=st.selectbox("RAM",df['Ram'].unique())


weight=st.number_input("Weight of the Laptop")
touchscreen=st.selectbox("Touchscreen",['No','Yes'])
ips=st.selectbox("Ips",['No','Yes'])
screen_size=st.number_input('Screen Size')
resolution=st.selectbox('Screen Resolution',['1920x1000','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])


cpu=st.selectbox('CPU',df['Cpu brand'].unique())
hdd=st.selectbox("HDD",[0,128,256,512,1024,2048])
ssd=st.selectbox("SSD",[0,8,128,256,512,1024])
gpu=st.selectbox("GPU",df['gpubrand'].unique())
os=st.selectbox("OS",df['os'].unique())
if st.button("Predict Price"):
    ppi=None

    if touchscreen== 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0
        
    if ips== 'Yes':
        ips= 1 
    else:
        ips= 0 
    x_res=int(resolution.split('x')[0])
    y_res=int(resolution.split('x')[1])
    ppi=((x_res**2) + (y_res**2))**0.5/screen_size
    query=np.array([company,type,ram,weight,touchscreen,ips,ppi,cpu,hdd,ssd,gpu,os])
    query=query.reshape(1,12)
    model=str(int(np.exp(pipe.predict(query)[0])))
    st.title("The Predicted Price of this configuration is " + model)
    