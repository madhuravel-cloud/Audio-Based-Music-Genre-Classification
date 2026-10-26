import gc
import os
import tempfile
import streamlit as st
from main import genre
from main2 import instrument

st.set_page_config(page_title="Music Analyzer", page_icon="🎵", layout="centered")

st.title("🎵 Music Analyzer")
st.write("Choose an audio file to predict its genre and instrument.")

uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg"])

if uploaded_file is not None:
    st.audio(uploaded_file)

    if st.button("Analyze Music", type="primary"):
        with st.spinner("Analyzing audio... Please wait."):
            suffix = os.path.splitext(uploaded_file.name)[1] or ".wav"
            path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
                    temp.write(uploaded_file.getvalue())
                    path = temp.name

                pred_genre = str(genre(path))
                gc.collect()
                pred_instrument = str(instrument(path))
                gc.collect()

                st.subheader("Results")
                st.success(f"**Genre:** {pred_genre}")
                st.info(f"**Instrument:** {pred_instrument}")

            except Exception as e:
                st.error(f"Could not analyze the audio: {e}")
            finally:
                if path and os.path.exists(path):
                    os.remove(path)