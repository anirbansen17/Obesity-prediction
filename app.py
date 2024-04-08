import streamlit as st
import os
import pickle
import streamlit as st
# from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd

# Load the machine learning model
mp = pickle.load(open('model_pickle.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

# Function to process input data and convert it into a numpy array
def process_input_data(gender, age, height, weight, family_overweight, favc, fcvc, ncp, caec, ch20, calc, smoke, faf, scc, mtrans, tue):

    # Convert categorical data to numerical data
    gender_female = 1 if gender == 'Female 👩' else 0
    gender_male = 1 if gender == 'Male 👨' else 0
    family_overweight_no = 1 if family_overweight == 'No' else 0
    family_overweight_yes = 1 if family_overweight == 'Yes' else 0
    favc_no = 1 if favc == 'No' else 0
    favc_yes = 1 if favc == 'Yes' else 0
    if fcvc == 'Never':
        fcvc = 1
    elif fcvc == 'Sometimes':
        fcvc = 2
    else:
        fcvc = 3 

    if ch20 == 'Less than a liter':
        ch20 = 1
    elif ch20 == 'Between 1 and 2 L':
        ch20 = 2
    else:
        ch20 = 3

    if faf == 'Zero':
        faf = 0
    elif faf == '1 or 2 days':
        faf = 1
    elif faf == '2 or 4 days':
        faf = 2
    else:
        faf = 3

    if tue == '0–2 hours':
        tue = 0
    elif tue == '3–5 hours':
        tue = 1
    else:
        tue = 2
    caec_always = 1 if caec == 'Always' else 0
    caec_frequently = 1 if caec == 'Frequently' else 0
    caec_sometimes = 1 if caec == 'Sometimes' else 0
    caec_no = 1 if caec == 'Never' else 0
    smoke_no = 1 if smoke == 'No' else 0
    smoke_yes = 1 if smoke == 'Yes' else 0
    scc_no = 1 if scc == 'No' else 0
    scc_yes = 1 if scc == 'Yes' else 0
    calc_always = 1 if calc == 'Always' else 0
    calc_frequently = 1 if calc == 'Frequently' else 0
    calc_sometimes = 1 if calc == 'Sometimes' else 0
    calc_no = 1 if calc == 'Never' else 0
    mtrans_automobile = 1 if mtrans == 'Automobile 🚗' else 0
    mtrans_bike = 1 if mtrans == 'Bike 🚲' else 0
    mtrans_motorbike = 1 if mtrans == 'Motorbike 🏍️' else 0
    mtrans_public_transportation = 1 if mtrans == 'Public Transportation 🚌' else 0
    mtrans_walking = 1 if mtrans == 'Walking🚶‍♂️' else 0
    
    
    # Convert input data to a numpy array
    input_data = np.array([age, height, weight, fcvc, ncp, ch20, faf, tue, gender_female, gender_male, family_overweight_no, family_overweight_yes, favc_no, favc_yes, caec_always, caec_frequently, caec_sometimes, caec_no, smoke_no, smoke_yes, scc_no, scc_yes, calc_always, calc_frequently, calc_sometimes, calc_no, mtrans_automobile, mtrans_bike, mtrans_motorbike, mtrans_public_transportation, mtrans_walking])
    
    return input_data.reshape(1, -1)


# predict function
def predict(input_data):
    prediction = mp.predict(input_data)
    prediction_label = ''
    if(prediction[0] == 0):
        prediction_label = "Underweight"
    elif(prediction[0] == 1):
        prediction_label = "Normal Weight"
    elif(prediction[0] == 2):
        prediction_label = "Obesity Type I"
    elif(prediction[0] == 3):
        prediction_label = "Obesity Type II"
    elif(prediction[0] == 4):
        prediction_label = "Obesity Type III"
    elif(prediction[0] == 5):
        prediction_label = "Overweight Level I"
    else:
        prediction_label = "Overweight Level II"
    return prediction_label


st.set_page_config(page_title="Obesity prediction",
                   layout="wide",
                   page_icon="🧑‍⚕️")

# Define your pages
def page1():
    # page title
    title_container = st.container()
    with title_container:
        st.title('Obesity Prediction')
        st.header('' , divider='rainbow')
        st.write('')
        st.write('')



    st.sidebar.subheader(':red[Red(High Risk)]')
    st.sidebar.error('**Alert:** Your recent health assessment indicates a **high risk** of obesity-related complications. Please consult with a healthcare professional immediately for a comprehensive evaluation and personalized advice.')

    st.sidebar.subheader(':orange[Yellow(Moderate Risk)]')
    st.sidebar.warning('**Caution:** You are currently at a **moderate risk** of developing obesity. We recommend increasing physical activity and seeking nutritional guidance to improve your health.')

    st.sidebar.subheader(':green[Green(Low Risk)]')
    st.sidebar.success('**Good News:** Your health indicators are within a healthy range, suggesting a **low risk** of obesity. Keep up the good lifestyle choices!')

    # st.sidebar.markdown(f'<h3 style="color:{color};">Obesity Prediction</h3>', unsafe_allow_html=True)
    # st.sidebar.markdown(f'<p style="color:{color};">Underweight: Less than 18.5</p>', unsafe_allow_html=True)
    # st.sidebar.markdown(f'<p style="color:{color};">Normal: 18.5 to 24.9</p>', unsafe_allow_html=True)
    # st.sidebar.markdown(f'<p style="color:{color};">Overweight: 25.0 to 29.9</p>', unsafe_allow_html=True)
    # st.sidebar.markdown(f'<p style="color:{color};">Obesity I: 30.0 to 34.9</p>', unsafe_allow_html=True)
    # st.sidebar.markdown(f'<p style="color:{color};">Obesity II: 35.0 to 39.9</p>', unsafe_allow_html=True)
    # st.sidebar.markdown(f'<p style="color:{color};">Obesity III: Higher than 40</p>', unsafe_allow_html=True)


    def predict_obesity(bmi):
        if bmi < 18.5:
            return "Underweight"
        elif 18.5 <= bmi <= 24.9:
            return "Normal"
        elif 25.0 <= bmi <= 29.9:
            return "Overweight"
        elif 30.0 <= bmi <= 34.9:
            return "Obesity I"
        elif 35.0 <= bmi <= 39.9:
            return "Obesity II"
        else:
            return "Obesity III"


    bmi = st.sidebar.slider("Enter your BMI:", min_value=10.0, max_value=50.0, step=0.1, value=22.0)

    obesity_category = predict_obesity(bmi)

    st.sidebar.write(f"Based on your BMI of {bmi}, your obesity category is: **{obesity_category}**")



    # MAIN PART

    # CONTAINER -1 : personol info

    # Create a container for the personal information inputs
    with st.container():
        st.subheader(':blue[Personal Information]')

        # Display input parameters for personal information
        col1, col2, col3 = st.columns(3 , gap='large')

        with col1:
            st.write('#### Gender')
            gender = st.selectbox('', ('Male 👨', 'Female 👩'),index=None,placeholder="Select your gender")

        with col2:
            st.write('#### Age')
            age = st.number_input('', min_value=0)

        with col3:
            st.write('#### Height (cm)')
            height = st.number_input('', min_value=0.0)

        with col1:
            st.write('#### Weight (kg)')
            weight_key = 'weight_key'
            weight = st.number_input('', min_value=0.0, key=weight_key)

        with col2:
            st.write('#### Family history of overweight')
            family_overweight = st.selectbox("", ('Yes', 'No '),index=None,
    placeholder="Choose an option",)


    st.write("")
    st.write("")

    # CONTAINER -2 : eating habits

    # Create a container for the eating habits inputs
    with st.container():
        st.subheader(':blue[Eating Habits] ')


        # Display input parameters for eating habits
        col1, col2 , col3 = st.columns(3 , gap="large")

        with col1:
            st.write('#### High Calorie Food Intake 🍜')    #Frequent consumption of high caloric food (FAVC)
            favc_key = 'favec_key'
            favc = st.selectbox('', ('Yes', 'No'),index=None , key=favc_key)

        with col2: 
            st.write('#### Frequent vegetable consumption 🥦')
            fcvc_key = 'fcvc_key'
            fcvc = st.selectbox('', ('Never', 'Sometimes', 'Always'),index=None, key=fcvc_key)

        with col3:
            st.write('#### Number of main meals 🍲')
            ncp = st.number_input('', min_value=1, max_value=10)

        with col1:
            st.write('#### Consumption of food btw meal 🍪 ')
            caec = st.selectbox('', ('Always', 'Frequently', 'Sometimes', 'Never') , index=None)

        with col2:
            st.write('#### Consumption of water daily 💧')
            ch20_key = 'ch20_key'  # Unique key for this radio button
            ch20 = st.selectbox('', ('Less than a liter', 'Between 1 and 2 L', 'More than 2 L'),index=None, key=ch20_key)

        with col3:
            st.write('#### Consumption of alcohol 🍺')
            calc_key = 'calc_key'  # Unique key for this radio button
            calc = st.selectbox('', ('Always', 'Frequently', 'Sometimes', 'Never'),index=None, key=calc_key)

        with col1:
            st.write('#### Smoking 🚬')
            smkg_key = 'smkg_key'  # Unique key for this radio button
            smkg = st.selectbox('', ('Yes', 'No'), index=None, key=smkg_key)




    st.write("")
    st.write("")

    # CONTAINER 3 - Physical Condition Attributes

    with st.container():
        st.subheader(':blue[Physical Condition]')

        # Divide the container into two columns
        col1, col2 , col3 = st.columns(3 , gap="large")

        with col1:
            # Physical activity frequency (FAF)
            st.write('#### Physical Activity Frequency 🏊‍♀️')
            faf_key = 'faf_key'  # Unique key for this radio button
            faf = st.selectbox('', ('Zero', '1 or 2 days', '2 or 4 days', '4 or 5 days'),index=None, key=faf_key)

        with col2:
            # Calories consumption monitoring (SCC)
            st.write('#### Calories Consumption Monitoring 📈')
            scc_key = 'scc_key'  # Unique key for this radio button
            scc = st.selectbox('', ('Yes', 'No'),index=None, key=scc_key)

        with col3:
            # Transportation used (MTRANS)
            st.write ('#### Transportation Used ')
            mtrans_key = 'mtrans_key'  # Unique key for this select box
            mtrans_options = ('Automobile 🚗', 'Motorbike 🏍️', 'Bike 🚲', 'Public Transportation 🚌', 'Walking🚶‍♂️')
            mtrans = st.selectbox('', mtrans_options, index=None, key=mtrans_key)

        
        with col1:
            # Time using technology devices (TUE)
            st.write('#### Time Using Technology Devices 🧑‍💻')
            tue_key = 'tue_key'  # Unique key for this radio button
            tue = st.selectbox('', ('0–2 hours', '3–5 hours', 'More than 5 hours'),index=None, key=tue_key)


    # Store variables in session state
    st.session_state['page1_variables'] = {
        'gender': gender,
        'age': age,
        'height': height,
        'weight': weight,
        'family_overweight': family_overweight,
        'favc': favc,
        'fcvc': fcvc,
        'ncp': ncp,
        'caec': caec,
        'ch20': ch20,
        'calc': calc,
        'smkg': smkg,
        'faf': faf,
        'scc': scc,
        'mtrans': mtrans,
        'tue': tue
    }

    
    if st.button("Predict Obesity Category"):
        st.session_state['page'] = 'page2'



def page2():

    st.sidebar.subheader(':red[Red(High Risk)]')
    st.sidebar.error('**Alert:** Your recent health assessment indicates a **high risk** of obesity-related complications. Please consult with a healthcare professional immediately for a comprehensive evaluation and personalized advice.')

    st.sidebar.subheader(':orange[Yellow(Moderate Risk)]')
    st.sidebar.warning('**Caution:** You are currently at a **moderate risk** of developing obesity. We recommend increasing physical activity and seeking nutritional guidance to improve your health.')

    st.sidebar.subheader(':green[Green(Low Risk)]')
    st.sidebar.success('**Good News:** Your health indicators are within a healthy range, suggesting a **low risk** of obesity. Keep up the good lifestyle choices!')

    # st.sidebar.markdown(f'<h3 style="color:{color};">Obesity Prediction</h3>', unsafe_allow_html=True)
    # st.sidebar.markdown(f'<p style="color:{color};">Underweight: Less than 18.5</p>', unsafe_allow_html=True)
    # st.sidebar.markdown(f'<p style="color:{color};">Normal: 18.5 to 24.9</p>', unsafe_allow_html=True)
    # st.sidebar.markdown(f'<p style="color:{color};">Overweight: 25.0 to 29.9</p>', unsafe_allow_html=True)
    # st.sidebar.markdown(f'<p style="color:{color};">Obesity I: 30.0 to 34.9</p>', unsafe_allow_html=True)
    # st.sidebar.markdown(f'<p style="color:{color};">Obesity II: 35.0 to 39.9</p>', unsafe_allow_html=True)
    # st.sidebar.markdown(f'<p style="color:{color};">Obesity III: Higher than 40</p>', unsafe_allow_html=True)


    def predict_obesity(bmi):
        if bmi < 18.5:
            return "Underweight"
        elif 18.5 <= bmi <= 24.9:
            return "Normal"
        elif 25.0 <= bmi <= 29.9:
            return "Overweight"
        elif 30.0 <= bmi <= 34.9:
            return "Obesity I"
        elif 35.0 <= bmi <= 39.9:
            return "Obesity II"
        else:
            return "Obesity III"


    bmi = st.sidebar.slider("Enter your BMI:", min_value=10.0, max_value=50.0, step=0.1, value=22.0)

    obesity_category = predict_obesity(bmi)

    st.sidebar.write(f"Based on your BMI of {bmi}, your obesity category is: **{obesity_category}**")



    # Retrieve variables from session state
    variables = st.session_state.get('page1_variables', {})

    # Review Section
    st.write("## Review Input:")
    # Personal Information
    st.subheader(':orange[Personal Information : ]')
    st.write(f"##### Gender : {variables.get('gender', '')}")
    st.write(f"##### Age : {variables.get('age', '')}")
    st.write(f"##### Height : {variables.get('height', '')} cm")
    st.write(f"##### Weight : {variables.get('weight', '')} kg")
    st.write(f"##### Family History of Overweight : {variables.get('family_overweight', '')}")

    # Eating Habits
    st.subheader(':orange[Eating Habits : ]')
    st.write(f"##### Frequent consumption of high caloric food (FAVC) :  {variables.get('favc', '')}")
    st.write(f"##### Frequency of consumption of vegetables (FCVC) : {variables.get('fcvc', '')}")
    st.write(f"##### Number of main meals (NCP) : {variables.get('ncp', '')}")
    st.write(f"##### Consumption of food between meals (CAEC) : {variables.get('caec', '')}")
    st.write(f"##### Consumption of water daily (CH20) : {variables.get('ch20', '')}")
    st.write(f"##### Consumption of alcohol (CALC) : {variables.get('calc', '')}")
    st.write(f"##### Smoking: {variables.get('smkg', '')}")

    # Physical Condition
    st.subheader(':orange[Physical Condition : ]')
    st.write(f"##### Calories consumption monitoring (SCC) : {variables.get('faf', '')}")
    st.write(f"##### Physical activity frequency (FAF) : {variables.get('scc', '')}")
    st.write(f"##### Time using technology devices (TUE) : {variables.get('mtrans', '')}")
    st.write(f"##### Transportation used (MTRANS) : {variables.get('tue', '')}")

    input_data = process_input_data(variables.get('gender'), variables.get('age'), variables.get('height'), variables.get('weight'), variables.get('family_overweight'), variables.get('favc'), variables.get('fcvc'), variables.get('ncp'), variables.get('caec'), variables.get('ch20'), variables.get('calc'), variables.get('smkg'), variables.get('faf'), variables.get('scc'), variables.get('mtrans'), variables.get('tue'))
    label = predict(input_data)
    if label == "Obesity Type I":
        st.error(' Prediction : Obesity Type I', icon="🚨")
    elif label == "Obesity Type II":
        st.error(' Prediction : Obesity Type II', icon="🚨")
    elif label == "Obesity Type III":
        st.error(' Prediction : Obesity Type III', icon="🚨")
    elif label == "Normal Weight":
        st.success(' Prediction : Normal Weight', icon="✅")
    elif label == "Overweight Level I":
        st.warning(' Prediction : Overweight Level I', icon="⚠️")
    elif label == "Overweight Level II":
        st.warning(' Prediction : Overweight Level II', icon="⚠️")
    else:
        st.error(' Prediction : Underweight', icon="🚨")

    if st.button("Go to Page 1"):
        st.session_state['page'] = 'page1'

# Set 'page1' as the default page
if 'page' not in st.session_state:
    st.session_state['page'] = 'page1'

# Display the current page
if st.session_state['page'] == 'page1':
    page1()
elif st.session_state['page'] == 'page2':
    page2()
