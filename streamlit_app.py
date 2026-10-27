import os
import tempfile
import librosa
import numpy as np
import pandas as pd
import streamlit as st
import main
import main2

st.set_page_config(page_title="Music Analyzer", layout="wide")

st.title("Music Analyzer & Dataset Builder")

if "analyzed" not in st.session_state:
    st.session_state.analyzed = False

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Upload & Librosa Analysis")
    uploaded_file = st.file_uploader("Select an audio file", type=["wav", "mp3", "ogg"])

    if uploaded_file is not None:
        st.audio(uploaded_file)

        if st.button("Analyze Audio", use_container_width=True):
            with st.spinner("Extracting features and predicting..."):
                suffix = os.path.splitext(uploaded_file.name)[1] or ".wav"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    path = tmp.name

                try:
                    y, sr = librosa.load(path, duration=30)
                    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                    cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
                    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
                    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))

                    pred_g, feat_g = main.genre(path)
                    pred_i, feat_i = main2.instrument(path)

                    step = max(1, len(y) // 1000)
                    st.session_state.waveform = y[::step]
                    st.session_state.stats = {
                        "Tempo (BPM)": round(float(np.mean(tempo)), 1),
                        "Centroid (Hz)": round(float(cent), 1),
                        "Rolloff (Hz)": round(float(rolloff), 1),
                        "Zero Crossing": round(float(zcr), 4)
                    }
                    st.session_state.pred_g = pred_g
                    st.session_state.pred_i = pred_i
                    st.session_state.feat_g = feat_g
                    st.session_state.feat_i = feat_i
                    st.session_state.filename = uploaded_file.name
                    st.session_state.analyzed = True
                finally:
                    if os.path.exists(path):
                        os.remove(path)

    if st.session_state.analyzed:
        st.write("**Librosa Audio Metrics**")
        m1, m2, m3, m4 = st.columns(4)
        stats = st.session_state.stats
        m1.metric("Tempo (BPM)", stats["Tempo (BPM)"])
        m2.metric("Centroid (Hz)", stats["Centroid (Hz)"])
        m3.metric("Rolloff (Hz)", stats["Rolloff (Hz)"])
        m4.metric("Zero Crossing", stats["Zero Crossing"])

        st.write("**Audio Waveform**")
        st.line_chart(pd.DataFrame({"Amplitude": st.session_state.waveform}), height=180)

with col2:
    st.subheader("2. Predictions & Add to Dataset")
    if st.session_state.analyzed:
        p1, p2 = st.columns(2)
        p1.info(f"**Predicted Genre:** {st.session_state.pred_g}")
        p2.info(f"**Predicted Instrument:** {st.session_state.pred_i}")

        st.divider()
        st.write("**Suggest Correct Labels & Save to Dataset**")
        st.caption("If the prediction is wrong or you want to add this song to the training data, confirm or type the label below.")

        genre_options = main.CLASSES + ["Add Custom..."]
        g_idx = genre_options.index(st.session_state.pred_g) if st.session_state.pred_g in genre_options else 0
        sel_g = st.selectbox("Correct Genre", genre_options, index=g_idx)
        final_g = st.text_input("Enter Custom Genre") if sel_g == "Add Custom..." else sel_g

        inst_options = main2.CLASSES + ["Add Custom..."]
        i_idx = inst_options.index(st.session_state.pred_i) if st.session_state.pred_i in inst_options else 0
        sel_i = st.selectbox("Correct Instrument", inst_options, index=i_idx)
        final_i = st.text_input("Enter Custom Instrument") if sel_i == "Add Custom..." else sel_i

        if st.button("Add Song to Dataset & Retrain", use_container_width=True):
            if final_g and final_i:
                with st.spinner("Updating CSV files and retraining models..."):
                    main.add_genre_data(st.session_state.feat_g, final_g, st.session_state.filename)
                    main2.add_instrument_data(st.session_state.feat_i, final_i, st.session_state.filename)
                st.success(f"Added '{st.session_state.filename}' ({final_g}, {final_i}) to datasets and retrained models!")
            else:
                st.warning("Please provide both a genre and an instrument label.")
    else:
        st.write("Upload and analyze an audio file on the left to view predictions and add to the dataset.")