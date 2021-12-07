import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ['price','area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom',
#        'basement', 'hotwaterheating', 'airconditioning', 'parking',
#        'prefarea']


def sns_plot(chart_type, df):
    '''Plot types are 
    Scatter plot,
    Histogram,
    Box plot,
    Boxen plot,
    Count plot,
    Bar plot
    '''
    
    sns.set_style(style='dark')
    
    # Scatter plot
    fig,ax = plt.subplots(figsize=(10,10))
    with st.sidebar.form("select-form"):
        if chart_type == "Scatter plot":
            selected_x_var = st.selectbox('''What do want the x variable to
                        be?''',
                    ['price','area', 'bedrooms', 'bathrooms', 'stories', 'parking'])
                        # y variable
            selected_y_var = st.selectbox('What about the y?',
                    ['price','area', 'bedrooms', 'bathrooms', 'stories', 'parking'])
            
            sns.scatterplot(data=df,x=selected_x_var,y=selected_y_var)
            ax.set_title(f"Scatter plot of {selected_x_var} vs {selected_y_var}")
            
        
        # Histogram plot
        elif chart_type =="Histogram":
            selected_x_var = st.selectbox('''What do want the x variable to
                        be?''',
                        ['price','area', 'bedrooms', 'bathrooms', 'stories', 'parking'])

            sns.histplot(data=df,x=selected_x_var,kde=True)
            # ax.set_title(f"Histogram plot of {selected_x_var}")
            ax.set_title(f"Skewness of {selected_x_var}: {np.around(df[selected_x_var].skew(axis=0),2)}")


        # Categorical plot 1: Box
        elif chart_type == "Box plot":
            selected_y_var = st.selectbox('''What do want the y variable to
                        be?''',
                        [ 'mainroad', 'guestroom',
                        'basement', 'hotwaterheating', 'airconditioning', 'prefarea'])

            selected_x_var = 'price'
            sns.boxplot(x=selected_x_var,y=selected_y_var, data=df,palette='nipy_spectral',hue=selected_y_var)
            ax.set_title(f"The Box plot of {selected_y_var}")
        
        # Categorical plot 2: Boxen
        elif chart_type == "Boxen plot":
            selected_y_var = st.selectbox('''What do want the y variable to
                        be?''',
                        [ 'mainroad', 'guestroom',
                        'basement', 'hotwaterheating', 'airconditioning', 'prefarea'])

            selected_x_var = 'price'
            sns.boxenplot(x=selected_x_var,y=selected_y_var, data=df,palette='nipy_spectral',hue=selected_y_var)
            ax.set_title(f"The Boxen plot of {selected_y_var}")
        
        # Categorical plot 3: Count
        elif chart_type == "Count plot":
            selected_x_var = st.selectbox('''What do want the x variable to
                        be?''',
                        [ 'mainroad', 'guestroom',
                        'basement', 'hotwaterheating', 'airconditioning', 'prefarea'])

            sns.countplot(data=df,x=selected_x_var,hue=selected_x_var)
            ax.set_title(f"The Count plot of {selected_x_var}")

        # Categorical plot 4: Bar 
        elif chart_type == "Bar plot":
            selected_x_var = st.selectbox('''What do want the x variable to
                        be?''',
                        [ 'mainroad', 'guestroom',
                        'basement', 'hotwaterheating', 'airconditioning', 'prefarea'])

            selected_y_var = 'price'
            sns.barplot(x=selected_x_var,y=selected_y_var, data=df,palette='nipy_spectral',hue=selected_x_var)
            ax.set_title(f"The Bar plot of {selected_x_var}")
        st.form_submit_button()

    return fig


# Log vs normal plot
def comparsion_plot_log(df_1):
    col1,col2=st.columns([10,10])
    with st.container():
        with col1:
            fig1,ax1 = plt.subplots()
            x_var = 'price'
            ax1 = sns.boxplot(df_1[x_var],palette='nipy_spectral')
            ax1.set_title(f"Skewness: {np.around(df_1[x_var].skew(),3)}")
            st.pyplot(fig1)

        with col2:
            fig2,ax2 = plt.subplots()
            x_var = 'price_log'
            ax2 = sns.boxplot(df_1[x_var],palette='nipy_spectral')
            ax2.set_title(f"Skewness: {np.around(df_1[x_var].skew(),3)}")
            st.pyplot(fig2)

        with col1:
            fig3,ax3 = plt.subplots()
            x_var = 'area'
            ax3 = sns.boxplot(df_1[x_var],palette='nipy_spectral')
            ax3.set_title(f"Skewness: {np.around(df_1[x_var].skew(),3)}")
            st.pyplot(fig3)

        with col2:
            fig4,ax4 = plt.subplots()
            x_var = 'area_log'
            ax4 = sns.boxplot(df_1[x_var],palette='nipy_spectral')
            ax4.set_title(f"Skewness: {np.around(df_1[x_var].skew(),3)}")
            st.pyplot(fig4)

# flooring vs normal plot
def comparsion_plot_flooring(df_floor,df_norm):
    col1,col2=st.columns([10,10])
    with st.container():
        with col1:
            fig1,ax1 = plt.subplots()
            x_var = 'price'
            ax1 = sns.boxplot(df_floor[x_var],palette='nipy_spectral')
            ax1.set_title(f"Skewness after flooring: {np.around(df_floor[x_var].skew(),3)}")
            st.pyplot(fig1)

        with col2:
            fig2,ax2 = plt.subplots()
            x_var = 'price'
            ax2 = sns.boxplot(df_norm[x_var],palette='nipy_spectral')
            ax2.set_title(f"Skewness: {np.around(df_norm[x_var].skew(),3)}")
            st.pyplot(fig2)

        with col1:
            fig3,ax3 = plt.subplots()
            x_var = 'area'
            ax3 = sns.boxplot(df_floor[x_var],palette='nipy_spectral')
            ax3.set_title(f"Skewness afer flooring: {np.around(df_floor[x_var].skew(),3)}")
            st.pyplot(fig3)

        with col2:
            fig4,ax4 = plt.subplots()
            x_var = 'area'
            ax4 = sns.boxplot(df_norm[x_var],palette='nipy_spectral')
            ax4.set_title(f"Skewness: {np.around(df_norm[x_var].skew(),3)}")
            st.pyplot(fig4)


# IQR vs normal plot
def comparsion_plot_IQR(df_floor,df_norm):
    col1,col2=st.columns([10,10])
    with st.container():
        with col1:
            fig1,ax1 = plt.subplots()
            x_var = 'price'
            ax1 = sns.boxplot(df_floor[x_var],palette='nipy_spectral')
            ax1.set_title(f"Skewness after IQR: {np.around(df_floor[x_var].skew(),3)}")
            st.pyplot(fig1)

        with col2:
            fig2,ax2 = plt.subplots()
            x_var = 'price'
            ax2 = sns.boxplot(df_norm[x_var],palette='nipy_spectral')
            ax2.set_title(f"Skewness: {np.around(df_norm[x_var].skew(),3)}")
            st.pyplot(fig2)

        with col1:
            fig3,ax3 = plt.subplots()
            x_var = 'area'
            ax3 = sns.boxplot(df_floor[x_var],palette='nipy_spectral')
            ax3.set_title(f"Skewness afer IQR: {np.around(df_floor[x_var].skew(),3)}")
            st.pyplot(fig3)

        with col2:
            fig4,ax4 = plt.subplots()
            x_var = 'area'
            ax4 = sns.boxplot(df_norm[x_var],palette='nipy_spectral')
            ax4.set_title(f"Skewness: {np.around(df_norm[x_var].skew(),3)}")
            st.pyplot(fig4)

# Z-score vs normal plot
def comparsion_plot_Zscore(df_z,df_norm):
    col1,col2=st.columns([10,10])
    with st.container():
        with col1:
            fig1,ax1 = plt.subplots()
            x_var = 'price'
            ax1 = sns.boxplot(df_z[x_var],palette='nipy_spectral')
            ax1.set_title(f"Skewness after Z-score: {np.around(df_z[x_var].skew(),3)}")
            st.pyplot(fig1)

        with col2:
            fig2,ax2 = plt.subplots()
            x_var = 'price'
            ax2 = sns.boxplot(df_norm[x_var],palette='nipy_spectral')
            ax2.set_title(f"Skewness: {np.around(df_norm[x_var].skew(),3)}")
            st.pyplot(fig2)

        with col1:
            fig3,ax3 = plt.subplots()
            x_var = 'area'
            ax3 = sns.boxplot(df_z[x_var],palette='nipy_spectral')
            ax3.set_title(f"Skewness afer Z-score: {np.around(df_z[x_var].skew(),3)}")
            st.pyplot(fig3)

        with col2:
            fig4,ax4 = plt.subplots()
            x_var = 'area'
            ax4 = sns.boxplot(df_norm[x_var],palette='nipy_spectral')
            ax4.set_title(f"Skewness: {np.around(df_norm[x_var].skew(),3)}")
            st.pyplot(fig4)

# Median vs normal plot
def comparsion_plot_median(df_median,df_norm):
    col1,col2=st.columns([10,10])
    with st.container():
        with col1:
            fig1,ax1 = plt.subplots()
            x_var = 'price'
            ax1 = sns.boxplot(df_median[x_var],palette='nipy_spectral')
            ax1.set_title(f"Skewness after use of median: {np.around(df_median[x_var].skew(),3)}")
            st.pyplot(fig1)

        with col2:
            fig2,ax2 = plt.subplots()
            x_var = 'price'
            ax2 = sns.boxplot(df_norm[x_var],palette='nipy_spectral')
            ax2.set_title(f"Skewness: {np.around(df_norm[x_var].skew(),3)}")
            st.pyplot(fig2)

        with col1:
            fig3,ax3 = plt.subplots()
            x_var = 'area'
            ax3 = sns.boxplot(df_median[x_var],palette='nipy_spectral')
            ax3.set_title(f"Skewness after use of median: {np.around(df_median[x_var].skew(),3)}")
            st.pyplot(fig3)

        with col2:
            fig4,ax4 = plt.subplots()
            x_var = 'area'
            ax4 = sns.boxplot(df_norm[x_var],palette='nipy_spectral')
            ax4.set_title(f"Skewness: {np.around(df_norm[x_var].skew(),3)}")
            st.pyplot(fig4)
       


# Heat map plot
def heat_map(df_heat):
    sns.set_theme()
    grid_kws = {"height_ratios": (1,.05),"hspace":.3}
    fig,(ax, cbar_ax) =plt.subplots(2,figsize=(15, 9),gridspec_kw=grid_kws)
    sns.heatmap(df_heat.corr(),
            ax=ax,
            cbar_ax=cbar_ax,
            cmap="rainbow",
            annot=True,
            vmin=-1,
            vmax=1,
            cbar_kws={"orientation":"horizontal",'label':"Christmas colorbar"},
            linewidths=1
            )
    return fig