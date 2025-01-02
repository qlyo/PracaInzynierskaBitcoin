import streamlit as st
from pathlib import Path
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project

# Uzyskanie ścieżki do katalogu głównego projektu
project_path = Path(__file__).resolve().parent

# Inicjalizacja projektu Kedro
bootstrap_project(project_path)

# Tworzenie sesji Kedro
with KedroSession.create(project_path=".") as session:
    context = session.load_context()

# Funkcja do wczytywania danych z katalogu Kedro
def load_data(dataset_name):
    return context.catalog.load(dataset_name)

# Aplikacja Streamlit
st.title("Przewidywanie cen Bitcoina")

if st.button("Załaduj dane"):
    data = load_data("btc_preprocessed_data")
    st.write("Dane przetworzone:")
    st.dataframe(data)
