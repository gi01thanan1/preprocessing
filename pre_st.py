import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import time 
# Pre-Processing
from sklearn.model_selection import train_test_split # train-test-split
from sklearn.impute import SimpleImputer, KNNImputer # detect & handle NaNs
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, OneHotEncoder # Ordinal Encoding, Nominal Encoding
from category_encoders import BinaryEncoder # Nominal Encoding 
from imblearn.under_sampling import RandomUnderSampler # undersampling
from imblearn.over_sampling import RandomOverSampler, SMOTE # oversampling
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler # Scaling

st.set_page_config(page_title="Healthcare!!!", page_icon=":bar_chart:",layout="wide")

hide_streamlit_style = """

    <style>
    .css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob, .styles_viewerBadge__1yB5_, .viewerBadge_link__1S137, .viewerBadge_text__1JaDK{ display: none; } #MainMenu{ visibility: hidden; } footer { visibility: hidden; } header { visibility: hidden; }
    
    
    </style>
    """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


st.title(" :bar_chart: Healthcare Stroke dataset EDA and preprocessing")
st.markdown('<style>div.block-container{padding-top:3rem;}</style>',unsafe_allow_html=True)
df = pd.read_csv('healthcare-dataset-stroke-data.csv')
#st.write(df)
st.write(df.style.background_gradient(cmap="Blues"))

with st.expander("features meaning"):
    st.write("id: unique identifier"  ) 
    st.write("gender: Male, Female or Other"  ) 
    st.write("age: age of the patient"  )
    st.write("hypertension: 0 if the patient doesn't have hypertension, 1 if the patient has hypertension"  ) 
    st.write("ever_married: No or Yes" ) 
    st.write("heart_disease: 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease"  ) 
    st.write("work_type: children, Govt_jov, Never_worked, Private or Self-employed"  ) 
    st.write("Residence_type: Rural or Urban"  )  
    st.write("avg_glucose_level: average glucose level in blood"  )
    st.write("bmi: body mass index is a value derived from the mass and height of a person. The BMI is defined as the body mass divided by the square of the body height, and is expressed"  )
    st.write("smoking_status: formerly smoked, never smoked, smokes or Unknown"  )
    st.write("stroke: 1 if the patient had a stroke or 0 if not"  )
    st.write("Note: 'Unknown' in smoking_status means that the information is unavailable for this patient"  )

df_stroke= pd.read_csv('df_stroke.csv')
df_stroke = df_stroke.drop(df_stroke.columns[0], axis=1)
#st.write(df_stroke)
with st.expander("data sample after extract features(body_mass_status , glucose_status) and handling unknown value in smoking status"):
    df_sample = df_stroke[0:5][["gender","age","hypertension","ever_married","heart_disease","work_type","Residence_type","smoking_status","body_mass_status","glucose_status","stroke"]]
    fig = ff.create_table(df_sample, colorscale = "Cividis")
    st.plotly_chart(fig)

with st.expander("Univariate Analysis"):
    num_cols = list(df_stroke.select_dtypes(include='number').columns)
    num_choice = st.radio(label="Numerical Column", options=["age","hypertension","heart_disease","bmi","avg_glucose_level","stroke"])
    
    uni_variant_num = st.button("Graph",key="num")
    if uni_variant_num == True:
        fig, axes = plt.subplots(nrows=1, ncols=2)
        sns.histplot(df_stroke[num_choice], kde=True, ax=axes[0])
        sns.boxplot(df_stroke[num_choice], ax=axes[1])
        st.pyplot(plt)
        #st.plotly_chart(fig)

    cat_cols = list(df_stroke.select_dtypes(include='O').columns)
    cat_choice = st.radio(label="Categorical Column", options=cat_cols)
    uni_variant_cat = st.button("Graph",key="cat")
    if uni_variant_cat == True:
        if df_stroke[cat_choice].nunique() <= 7:
            plt.figure(figsize=(12, 6))
            df_pie = df_stroke.groupby(cat_choice)[["age"]].count().sort_values(by='age', ascending=False)
            plt.pie(labels=df_pie.index, x=df_pie['age'], autopct="%1.1f%%")
        elif df_stroke[cat_choice].nunique() > 7 and df_stroke[cat_choice].nunique() < 35:
            #plt.figure(figsize=(10, 10))
            sns.countplot(y=df_stroke[cat_choice])
        else: # count > 35 show top 20
            #plt.figure(figsize=(10, 10))
            df_bar = df_stroke.groupby(cat_choice)[['age']].count().reset_index().sort_values(by='age', ascending=False).head(20)
            sns.barplot(y=df_bar[cat_choice], x=df_bar['age'])
        #plt.figure(figsize=(3, 2))
        #fig = plt. figure(figsize=(7,3)) 
        
        # fig, ax = plt.subplots(figsize=(12, 6))
        # #st.pyplot(fig, figsize=(5, 5))
        # st.plotly_chart(fig)
        st.pyplot(plt)
with st.expander("Bivariate Analysis"):
    graph=['stroke with age','hypertension with age','Percentage of People with stroke in each gender','Impact of overweight on People with stroke']
    graph_choice = st.selectbox(label="Choose graph", options=graph)

    def func1():
        
        plt.figure(figsize=(12, 6))
        sns.scatterplot(data=df_stroke,x="age",y="stroke",hue="gender")
        plt.title("stroke with age")
        plt.xlabel('age')
        plt.ylabel('stroke')
        
        st.pyplot(plt,use_container_width=True)
    def func2():
        plt.figure(figsize=(12, 6))
        sns.scatterplot(data=df_stroke,x="age",y="hypertension",hue="gender")
        plt.title("hypertension with age")
        plt.xlabel('age')
        plt.ylabel('hypertension')
        st.pyplot(plt)
       
    def func3():
        plt.figure()
        df_stroked = df_stroke[df_stroke['stroke'] == 1]
        
        df_pie = df_stroked.groupby('gender')[["age"]].count().sort_values(by='age', ascending=False)
        plt.pie(labels=df_pie.index, x=df_pie['age'], autopct="%1.1f%%")
        plt.title("Percentage of People with stroke in each gender")
        plt.show()
        st.pyplot(plt,use_container_width=False)
        
    def func4():
        df_stroked = df_stroke[df_stroke['stroke'] == 1]

        plt.figure(figsize=(10, 10))
        sns.countplot(y=df_stroked['body_mass_status'])
        plt.title("Impact of overweight on People with stroke")
        st.pyplot(plt)

    draw_graph = {'stroke with age':func1, "hypertension with age":func2,"Percentage of People with stroke in each gender":func3,"Impact of overweight on People with stroke":func4}
    draw_graph[graph_choice]()
with st.expander("multi_variate Analysis"):
    plot = sns.pairplot(df_stroke)
 
    # Display the plot in Streamlit
    st.pyplot(plot.fig)
with st.expander("preprocessing"):
    # train test split
    
    X = df_stroke.drop(['stroke'], axis=1)
    y = df_stroke['stroke']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y, random_state=42)
    #st.subheader('X_train after train_test_split:')
    st.markdown("<h3 style='color:red;'>X_train after train_test_split</h3>", unsafe_allow_html=True)
    st.write(X_train)
    # handeling nans in body_mass_status(fill None values with Nan )

    X_train[['body_mass_status']] = X_train[['body_mass_status']].fillna(value=np.nan)
    X_test[['body_mass_status']] = X_test[['body_mass_status']].fillna(value=np.nan)
    #  smoking_status, body_mass_status
    ## handle their NaNs from train and test with simpleimputer
    simple_imputer = SimpleImputer(strategy='most_frequent')
    X_train[['smoking_status', 'body_mass_status']] = simple_imputer.fit_transform(X_train[['smoking_status', 'body_mass_status']])
    X_test[['smoking_status', 'body_mass_status']] = simple_imputer.transform(X_test[['smoking_status', 'body_mass_status']])
    # handle 'bmi' with KNNImputer
    knn_imputer = KNNImputer(n_neighbors = 2)
    X_train[['bmi']] = knn_imputer.fit_transform(X_train[['bmi']])
    X_test[['bmi']] = knn_imputer.transform(X_test[['bmi']])
    #st.subheader('Check nan values after handling:')
    st.markdown("<h3 style='color:red;'>Check nan values after handling</h3>", unsafe_allow_html=True)
    for j in X_train.columns:
        no_null = X_train[j].isna().sum()
        #st.markdown(f"check number of null values : {no_null})
        st.markdown(f"no of null values in {j}: {no_null}")
    #st.markdown(f'<p class="custom-markdown">Guess the fruit name , the word is: {n} characters</p>', unsafe_allow_html=True,help="Enter only one character then press add  , continue entering and pressing to get the complete word")
    # d) Detect & Handle Outliers
    # bmi , avg_glucose_level are right skewed
    outlier_cols = ['bmi', 'avg_glucose_level']
    for col in outlier_cols:
        X_train[col] = np.log(X_train[col])
        X_test[col] = np.log(X_test[col])
    # Encoding
    #nomenal : gender		ever_married	work_type	Residence_type		smoking_status   body_mass_status	glucose_status
    #ordinal : 	
    ohe_encoder = OneHotEncoder(sparse_output=False, drop='first')
    result_train = ohe_encoder.fit_transform(X_train[['gender', 'ever_married','work_type','Residence_type','smoking_status','glucose_status']])
    result_test = ohe_encoder.transform(X_test[['gender', 'ever_married','work_type','Residence_type','smoking_status','glucose_status']])
    ohe_train_df = pd.DataFrame(result_train, columns=ohe_encoder.get_feature_names_out(), index=X_train.index)
    ohe_test_df = pd.DataFrame(result_test, columns=ohe_encoder.get_feature_names_out(), index=X_test.index)
    # binary encoding for 'body_mass_status'
    bin_encoder = BinaryEncoder()
    bi_train_df = bin_encoder.fit_transform(X_train[['body_mass_status']])
    bi_test_df = bin_encoder.transform(X_test[['body_mass_status']])
    # concat
    X_train = pd.concat([X_train, ohe_train_df, bi_train_df], axis=1).drop(['gender', 'ever_married','work_type','Residence_type','smoking_status','glucose_status','body_mass_status'], axis=1)
    X_test = pd.concat([X_test, ohe_test_df, bi_test_df], axis=1).drop(['gender', 'ever_married','work_type','Residence_type','smoking_status','glucose_status','body_mass_status'], axis=1)
    # apply smote oversampling on x_train , y_train
    smote = SMOTE(k_neighbors=2, random_state=42)

    X_train_resampled_smote, y_train_resampled_smote =  smote.fit_resample(X_train, y_train)
    # standered scaler for age
    std_scaler = StandardScaler()
    X_train_resampled_smote_scaled = X_train_resampled_smote.copy()
    x_test_scaled = X_test.copy()
    X_train_resampled_smote_scaled[['age']] = std_scaler.fit_transform(X_train_resampled_smote_scaled[['age']])
    x_test_scaled[['age']] = std_scaler.transform(x_test_scaled[['age']])
    # robust scaler for bmi and avg_glucose_level
    rbst_scaler = RobustScaler()
    X_train_resampled_smote_scaled[['bmi', 'avg_glucose_level']] = rbst_scaler.fit_transform(X_train_resampled_smote_scaled[['bmi', 'avg_glucose_level']])
    x_test_scaled[['bmi', 'avg_glucose_level']] = rbst_scaler.transform(x_test_scaled[['bmi', 'avg_glucose_level']])
    #st.subheader("Final x_train after preprocessing:")
    st.markdown("<h3 style='color:red;text-align:center;'>Final x_train after preprocessing</h3>", unsafe_allow_html=True)
    st.write(X_train_resampled_smote_scaled)
    # Download 4 files after preprocessing
    downl = st.button("Download 4 files",key="dl")
    if downl== True:
        X_train_resampled_smote_scaled.to_csv("X_train_scaled_resampled_scaled.csv")
        x_test_scaled.to_csv("X_test_scaled.csv")
        y_train_resampled_smote.to_csv("y_train_resampled.csv")
        y_test.to_csv("y_test.csv")

    
    

    

    