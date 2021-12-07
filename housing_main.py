import streamlit as st
import pandas as pd
import numpy as np
import warnings
from scipy import stats
from make_plots import *
import pickle
warnings.filterwarnings("ignore")

# st.set_page_config(layout="wide")

df = pd.read_csv("housing-data.zip")

# Copied the data
df_copy = df.copy()

df_copy = (
    df_copy
    .assign(
        price=df_copy.price.astype(np.int32),
        area=df_copy.area.astype(np.int16),
        bedrooms=df_copy.bedrooms.astype(np.int8),
        bathrooms=df_copy.bathrooms.astype(np.int8),
        stories=df_copy.stories.astype(np.int8),
        parking=df_copy.parking.astype(np.int8),
        mainroad=df_copy.mainroad.astype('category'),
        guestroom=df_copy.guestroom.astype('category'),
        basement=df_copy.basement.astype('category'),
        hotwaterheating=df_copy.hotwaterheating.astype('category'),
        airconditioning=df_copy.airconditioning.astype('category'),
        prefarea=df_copy.prefarea.astype('category'),
        furnishingstatus=df_copy.furnishingstatus.astype('category'),
    )
)


a_object = ['mainroad',
 'guestroom',
 'basement',
 'hotwaterheating',
 'airconditioning',
 'prefarea']

def binary_mapping(x):
    return x.map({'yes': 1, "no": 0})

df_copy[a_object] = df_copy[a_object].apply(binary_mapping)

# Dropping the first column because only two columns are needed
furnishing_status =  pd.get_dummies(df_copy['furnishingstatus'], drop_first=True)

# Adding columns into it and not rows; rows would be axis=0 or axis="rows"
housing = pd.concat([df_copy,furnishing_status],axis="columns")

housing.drop(columns="furnishingstatus", inplace=True)

st.title('Housing prediction App')
st.sidebar.subheader('An App by Daniel Ihenacho')
st.sidebar.write(('''This app  makes use of a housing dataset to make predictions about the price,
when the variables are adjusted. '''))
st.sidebar.markdown('---')


df1 = pd.DataFrame(housing.describe())


# display data
show_data = st.sidebar.checkbox("See the raw data and data description?",key='data')
with st.container():
    if show_data:
            housing
            df1

# Performing EDA
plot_types= ["Scatter plot",
    "Histogram",
    "Box plot",
    "Boxen plot",
    "Count plot",
    "Bar plot"]

# User choose type
plot_data = st.sidebar.checkbox("Do you want to visualise the data?", key="plot_data")
if plot_data:
    chart_type = st.sidebar.selectbox("Choose your chart type", plot_types)

    def show_plot(chart_type,data):
        # This calls the sns_plot method in make_plots
        # Which returns a figure i.e fig
        plot = sns_plot(chart_type,data)
        col_1,col_2 = st.columns((10,3))
        with col_1:
            st.pyplot(plot)
            plt.tight_layout()
    

    show_plot(chart_type,housing)



# Correcting Skewness
b_not = ['price', 'area', 'bedrooms', 'bathrooms', 'stories', 'parking']

# Log approach
housing_log = housing.copy()

fig, ax = plt.subplots(2,3, figsize=(20,10), constrained_layout=True)
ax = ax.ravel()


for value in (b_not):
    log = (f'{value}_log')
    housing_log[log] = housing_log[value].apply(lambda x: np.log(x+1))


# Quantile-based Flooring and Capping
housing_qfc = housing.copy()
housing_qfc['price'].where(housing_qfc['price']<7350000.0,7350000.0,inplace=True)
housing_qfc['area'].where(housing_qfc['area']<7980.0,7980.0,inplace=True)



# Use of IQR score
housing_iqr_score = housing.copy()
Q_1 = housing_iqr_score.price.quantile(0.25)
Q_3 = housing_iqr_score.price.quantile(0.75)
IQR = Q_3 - Q_1 
IQR_treatment = ((housing_iqr_score.price <= (Q_1 - (1.5*IQR))) | (housing_iqr_score.price >= (Q_3 + (1.5*IQR))))
housing_iqr_score = housing_iqr_score[~IQR_treatment]

Q_1 = housing_iqr_score.area.quantile(0.25)
Q_3 = housing_iqr_score.area.quantile(0.75)
IQR = Q_3 - Q_1 
IQR_treatment = ((housing_iqr_score.area <= (Q_1 - (1.5*IQR))) | (housing_iqr_score.area >= (Q_3 + (1.5*IQR))))
housing_iqr_score = housing_iqr_score[~IQR_treatment]

# Using Z_score 
housing_z = housing.copy()
z = np.abs(stats.zscore(housing_z.price))
# If a data point, df > 3 std,  then it is an outlier
threshold = 3
housing_z = housing_z[z<3]
z = np.abs(stats.zscore(housing_z.area))
housing_z = housing_z[z<3]


# Using median values
housing_median = housing.copy()
housing_median['price'].where(housing_median['price']<8400000.0,4340000.0,inplace=True)
housing_median['area'].where(housing_median['area']<9000.0,4600.0,inplace=True)



correct_data = st.sidebar.checkbox("Skewness correction types", key="correct_data")
if correct_data:
    st.write("""Levels of skewness\n
    1. (-0.5,0.5) = lowly skewed\n
    2. (-1,0-0.5) U (0.5,1) = Moderately skewed\n
    3. (-1 & beyond ) U (1 & beyond) = Highly skewed""")
    # with st.sidebar.form("log-p"):
    form = st.sidebar.form("log-p")
    menu = form.radio('Skewness treatment',options=("Log",'Floor',"IQR score",
    "Z score","Median"))
    if menu == 'Log':
        st.write("Log approach")
        comparsion_plot_log(housing_log)
    if menu == 'Floor':
        st.write("Quantile-based Flooring and Capping")
        comparsion_plot_flooring(housing_qfc,housing)
    if menu == 'IQR score':
        st.write("IQR score")
        comparsion_plot_IQR(housing_iqr_score,housing)
    if menu == 'Z score':
        st.write("Z score")
        comparsion_plot_IQR(housing_z,housing)
    if menu == 'Median':
        st.write("Median")
        comparsion_plot_median(housing_median,housing)
    form.form_submit_button()


# Building the Model 
plot_heat = st.sidebar.checkbox("The Model", key="plot_heat")
if plot_heat:
    st.write("""
    For this model, the quantile-based flooring and capping was used
    """)
    
    col1,col2 = st.columns((30,7))
    with col1:
        with st.expander("The heat map"):
            heatting = heat_map(housing_qfc)
            st.pyplot(heatting)
    with col2: 
        st.write("""It seems semi-furnished and unfurnished add no value to the data set, therefore, they can be dropped.""")
    
    with col1:
        # with st.expander('The model'):
        uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])
        if uploaded_file is not None:
            input_df = pd.read_csv(uploaded_file)
            input_df_coded = input_df.copy()
            input_df_coded[a_object] = input_df_coded[a_object].apply(binary_mapping)
        else:
            def user_input_features():
                
                with st.sidebar.form('user_inputs'):
                    with st.expander("User inputs"):
                        area = st.number_input("Area", min_value=0, value=7300)
                        bedrooms = st.number_input("Bedrooms",min_value=0, value=5)
                        bathrooms = st.number_input("Bathrooms",min_value=0, value=5)
                        stories = st.number_input("Stories",min_value=1, value=2)
                        mainroad = st.selectbox("Main road", options=["yes","no"])
                        guestroom = st.selectbox("Guest room", options=["yes","no"])
                        basement = st.selectbox("Basement", options=["yes","no"])
                        hotwater_h = st.selectbox("Hot-water heating", options=["yes","no"])
                        air_cond = st.selectbox("Airconditioning",options=["yes","no"])
                        parking = st.number_input("Parking",min_value=1, value=2)
                        prefarea = st.selectbox("Prefarea",options=["yes","no"])

                
                        if mainroad == "yes":
                            mainroad = 1
                        else:
                            mainroad = 0

                        # guest room
                        if guestroom == "yes":
                            guestroom = 1
                        else:
                            guestroom = 0

                        # Basement
                        if basement == "yes":
                            basement = 1
                        else:
                            basement = 0

                        # Hot-water heating
                        if hotwater_h == "yes":
                            hotwater_h = 1
                        else:
                            hotwater_h = 0

                        # Airconditioning
                        if air_cond == "yes":
                            air_cond = 1
                        else:
                            air_cond = 0
                        # Prefarea
                        if prefarea == "yes":
                            prefarea = 1
                        else:
                            prefarea = 0
                    st.form_submit_button()

                    data = {
                            "area": area,
                            "bedrooms ": bedrooms ,
                            "bathrooms ": bathrooms ,
                            "stories": stories,
                            "mainroad":mainroad,
                            "guestroom ": guestroom,
                            "basement ": basement,
                            "hotwaterheating ": hotwater_h,
                            "airconditioning ": air_cond,
                            "parking ": parking,
                            "prefarea ": prefarea
                            }
                    features = pd.DataFrame(data, index=[0])
                    return features

            input_df = user_input_features()


    with col1:
        if uploaded_file is not None:
            st.write(input_df)
            
        else:
            st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
            st.write(input_df)
        
        # Making use of the exported algorithm
        load_housing = pickle.load(open('housing.pkl', 'rb'))
        
        # Apply model to make predictions
        if uploaded_file is not None:
            prediction = load_housing.predict(input_df_coded)    
        else:
            prediction = load_housing.predict(input_df)
        data = {
            "price": prediction
        }
        price_out = pd.DataFrame(data)
        st.write(price_out)
        
        
        # if uploaded_file is not None:
        @st.cache
        def convert_df(df):
            return df.to_csv().encode('utf-8')
        csv = convert_df(price_out)

        st.download_button(
            label="Download your prediction",
            data=csv,
            file_name="predictions.csv",

        )