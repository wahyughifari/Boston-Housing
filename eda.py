import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def run():
    st.title("🔎 Exploratory Data Analysis - Boston Housing")

    @st.cache_data
    def load_data():
        df = pd.read_csv("boston_housing_data.csv")
        return df

    df = load_data()

    if st.checkbox("Show Raw Dataset"):
        st.dataframe(df)

    st.subheader("Distribution of Median House Prices (medv)")
    fig1, ax1 = plt.subplots()
    sns.histplot(df["medv"], kde=True, ax=ax1, color="skyblue")
    ax1.set_title("Distribution of Median House Prices")
    st.pyplot(fig1)

    if st.checkbox("Show Correlation Heatmap"):
        st.subheader("Correlation Heatmap")
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f", ax=ax2)
        st.pyplot(fig2)

    if st.checkbox("Show Room vs Price Scatterplot"):
        st.subheader("Average Rooms vs Median Price")
        fig3, ax3 = plt.subplots()
        sns.scatterplot(data=df, x="rm", y="medv", ax=ax3)
        ax3.set_title("Number of Rooms vs Median House Price")
        st.pyplot(fig3)

    if st.checkbox("Show Boxplot of 'chas' vs 'medv'"):
        st.subheader("Does proximity to Charles River affect house price?")
        fig4, ax4 = plt.subplots()
        sns.boxplot(data=df, x="chas", y="medv", ax=ax4)
        ax4.set_xticklabels(["Not Near River", "Near River"])
        ax4.set_title("Charles River Proximity vs Price")
        st.pyplot(fig4)

    if st.checkbox("Show Dataset Summary"):
        st.subheader("Dataset Summary")
        st.write(df.describe())

    st.markdown("""
    ---
    Created by Muhammad Wahyu Ghifari
    """)
