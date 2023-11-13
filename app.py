import streamlit as st

from pages import dashboard


def app():
    st.title("Marketing Mix Modeling")
    st.sidebar.markdown("### 機械学習に用いるcsvファイルを入力してください")
    upload_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
    if upload_file:
        dashboard.display(upload_file)


if __name__ == "__main__":
    app()
