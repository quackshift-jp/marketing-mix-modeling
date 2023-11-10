import streamlit as st

from pages import dashboard


def app():
    st.title("Marketing Mix Modeling")
    dashboard.display()


if __name__ == "__main__":
    app()
