# app.py

import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, WebRtcStreamerState
import av, cv2, pandas as pd, numpy as np, io, time, os
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

from src.detection.typing_activity import TypingActivityDetector
from src.gaze.face_mesh import FaceMeshDetector
from src.gaze.blink_detector import BlinkDetector
from src.gaze.combined_gaze import CombinedGazeDetector
from src.gaze.smoother import DirectionSmoother
from src.logic.focus_manager import FocusManager


st.set_page_config(page_title="Smart Focus AI", layout="wide")
st.title("🎯 :rainbow[SMART FOCUS AI]")
st.warning("Don't forget to allow the webcam access in your browser settings.")
st.markdown("""
    <a href="#live">
        <button style='font-size:16px;padding:10px 20px;margin-top:20px;'>🚀 Try the real-time detection now by clicking START</button>
    </a>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    video {
        max-width: 150px !important;
        height: auto !important;
        margin: 10px auto;
        display: block;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    video:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 20px rgba(0,0,0,0.2);
    }
    </style>
""", unsafe_allow_html=True)

# -- Paramètres de journalisation --
LOG_DIR = "Logs"
os.makedirs(LOG_DIR, exist_ok=True)
today = pd.Timestamp.now().date()
FILENAME_CSV = os.path.join(LOG_DIR, f"focus_logs_{today}.csv")

# Initialisation du DataFrame en session
if "df_log" not in st.session_state:
    st.session_state.df_log = pd.DataFrame(columns=["timestamp","focused","typing","gaze","blink_count"])
if "recording" not in st.session_state:
    st.session_state.recording = False

def make_pdf(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        fig, ax = plt.subplots(figsize=(6,4))
        focused_rate = df["focused"].mean() * 100
        blinks = df["blink_count"].max()
        typing_rate = df["typing"].mean() * 100
        gaze_rate = df["gaze"].mean() * 100
        stats_text = (
            f"Taux de focus : {focused_rate:.1f}%\n"
            f"Clignements : {blinks}\n"
            f"Taux typing : {typing_rate:.1f}%\n"
            f"Taux gaze   : {gaze_rate:.1f}%"
        )
        ax.text(0.1, 0.5, stats_text, fontsize=14)
        ax.axis("off")
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8,4))
        df_plot = df.set_index("timestamp")[["focused","typing","gaze"]]
        df_plot.rolling(30).mean().plot(ax=ax)
        ax.set_ylabel("Moyenne glissante")
        ax.set_title("Évolution Focus/Typing/Gaze")
        pdf.savefig(fig)
        plt.close(fig)

    return buf.getvalue()




class FocusTransformer(VideoProcessorBase):
    def __init__(self):
        self.typing = TypingActivityDetector(display_timeout=0.5)
        self.typing.start()
        self.face_mesh = FaceMeshDetector()
        self.blink = BlinkDetector()
        self.gaze = CombinedGazeDetector(self.face_mesh)
        self.focus_mgr = FocusManager(typing_detector=self.typing, gaze_detector=self.gaze)
        self.h_smoother = DirectionSmoother(window_size=5)
        self.v_smoother = DirectionSmoother(window_size=5)
        # Durée d’un cycle (affiché + caché) en secondes
        self.cycle = 1.0  

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        lm = self.face_mesh.process(img)
        if lm is not None:
            self.blink.update(img, lm, self.face_mesh)
        is_typing = self.typing.is_typing()
        is_gazing = self.gaze.is_gazing(img)
        focused = self.focus_mgr.is_focused(img)
        ts = time.time()
        new_row = {
            "timestamp": ts,
            "focused": int(focused),
            "typing": int(is_typing),
            "gaze": int(is_gazing),
            "blink_count": self.blink.blink_count
        }
        pd.DataFrame([new_row]).to_csv(FILENAME_CSV, mode="a", header=False, index=False)
        status = "Focused" if focused else "Distracted"
        color = (0,255,0) if focused else (0,0,255)
        cv2.putText(img, status, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(img, f"Typing:{int(is_typing)}  Center Gaze:{int(is_gazing)}  Blink:{self.blink.blink_count}",
                    (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        
        # CV2 ne sachant pas aligner à droite tout seul -->
        # récupère taille du texte
        (text_w, text_h), baseline = cv2.getTextSize(
            "HELLO STREAMLIT",
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            thickness=2
        )

        # Calcul phase entre 0 et 1
        phase = (time.time() % self.cycle) / self.cycle

        # Si phase < 0.5 : on affiche, sinon on ne dessine pas → clignote
        if phase < 0.5:
            text = "HELLO STREAMLIT"
            (w, h), _ = cv2.getTextSize(text,
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        fontScale=1,
                                        thickness=2)
            margin = 10
            x = img.shape[1] - w - margin
            y = h + margin

            cv2.putText(img, text, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")
    
st.markdown("## 🧠 :rainbow[What is Smart Focus AI]")
st.markdown("""

**Type** : Application Web interactive  
**Technos** : Python · Streamlit · OpenCV · MediaPipe · Machine Learning léger

---

**Description :**  
Smart Focus AI est une application web intelligente qui analyse **l’attention** de l’utilisateur en temps réel via la **webcam**. Elle combine plusieurs détecteurs pour évaluer en continu l’état de concentration :

- ✅ Détection de frappe au clavier (*typing*)  
- 🎯 Suivi du regard (*center gaze / gaze away*)  
- 👁️ Détection des clignements (*blinks*)  
- 🔄 Fusion des signaux pour déterminer si l’utilisateur est **Focused** ou **Distracted**

L’interface fournit :
- 📽️ Un retour visuel en direct (overlay vidéo)  
- 📊 Un rapport automatique au format **PDF** et **CSV**  
- 🎓 Des vidéos explicatives intégrées pour guider l’utilisateur
""")

st.markdown("## 🎓 :rainbow[How to Use Smart Focus AI]")

with st.expander("🎓 How does Smart Focus AI work?"):
    st.markdown("""
    1. The application activates your webcam (with your permission).
    2. It continuously monitors:
       - 🤓 your gaze direction (center or away),
       - 🖐️ your typing activity,
       - 👁️ your blinking frequency.
    3. These signals are combined to determine whether you're **Focused** or **Distracted**.
    4. At the end of the session, you get a **clear and exportable report** (CSV + PDF).
    """)

st.info("Tutorial : Here are 3 short videos (5s) to help you understand how the system works:")
st.write(" - 📽️ These videos illustrate the behaviors the system detects (not live recordings).")

st.markdown('<div id="live"></div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["✍️ Typing Detection", "🎯 Center Gaze", "🙈 Gaze Away"])

with tab1:
    st.markdown("**Typing Detection:** When you type, the system tracks activity. You will see -> Focused, Typing:1, Center Gaze:1")
    st.video("static/tutorials/only_typing.mp4", format="video/mp4", start_time=0)

with tab2:
    st.markdown("**Center Gaze:** When you look at the screen, you're marked as focused. You will see -> Focused, Typing: 0, Center Gaze:1")
    st.video("static/tutorials/only_gaze_center.mp4", format="video/mp4", start_time=0)

with tab3:
    st.markdown("**Gaze Away:** When you look away from the screen, you're marked as distracted. You will see --> Distracted, Typing: 0, Center Gaze:0")
    st.video("static/tutorials/gaze_away.mp4", format="video/mp4", start_time=0)


webrtc_ctx = webrtc_streamer(
    key="smart-focus",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=FocusTransformer,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# --- Gestion de l'état caméra pour détecter l'arrêt ---
if webrtc_ctx:
    prev_state = st.session_state.get("prev_webrtc_state", None)
    current_state = webrtc_ctx.state

    if prev_state == "PLAYING" and current_state != "PLAYING":
        st.session_state["show_success"] = True
        st.session_state["success_start_time"] = time.time()

    st.session_state["prev_webrtc_state"] = current_state

# --- Affichage pendant que la caméra tourne ---
if webrtc_ctx and webrtc_ctx.state.playing:
    st.markdown("🔴 **Recording in progress...**", unsafe_allow_html=True)
    live_chart = st.empty()

    for _ in range(100):
        if not webrtc_ctx.state.playing:
            break
        df_live = st.session_state.df_log.copy()

        if not df_live.empty:
            if not np.issubdtype(df_live["timestamp"].dtype, np.datetime64):
                df_live["timestamp"] = pd.to_datetime(df_live["timestamp"], unit="s")

            now = pd.Timestamp.now()
            df_recent = df_live[df_live["timestamp"] > now - pd.Timedelta(seconds=120)]
            df_recent.set_index("timestamp", inplace=True)

            live_chart.line_chart(df_recent[["focused", "typing", "gaze"]])
        else:
            live_chart.info("⏳ Waiting for focus data...")
        time.sleep(0.5)

# --- Affichage après arrêt de la webcam ---
if st.session_state.get("show_success", False):
    elapsed = time.time() - st.session_state["success_start_time"]
    if elapsed < 10:
        st.success("✅ Analysis completed successfully")
    else:
        st.session_state["show_success"] = False

# --- Sidebar ---
st.sidebar.markdown("<h1 style='text-align: center;'>Home</h1>", unsafe_allow_html=True)



def format_duration(seconds: float) -> str:
    """Formate une durée en secondes en 'Xh Ym Zs' ou 'Ym Zs'."""
    hours, rem = divmod(int(seconds), 3600)
    mins, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {mins}m {secs}s"
    return f"{mins}m {secs}s"

if os.path.exists(FILENAME_CSV):
    with st.spinner("📊 Analyse des données en cours..."):
        df_today = pd.read_csv(
            FILENAME_CSV,
            header=None,
            names=["timestamp", "focused", "typing", "gaze", "blink_count"]
        )
    # 1) Conversion du timestamp
    df_today["timestamp"] = pd.to_datetime(df_today["timestamp"], unit="s")
    # 2) Tri chronologique + calcul des écarts (dt)
    df_today = df_today.sort_values("timestamp")
    df_today["dt"] = df_today["timestamp"].diff().dt.total_seconds().fillna(0)
    # 3) Durée totale
    total_secs = (df_today["timestamp"].iloc[-1] - df_today["timestamp"].iloc[0]).total_seconds()
    total_fmt  = format_duration(total_secs)

    # 4) Durées effectives
    focus_secs      = df_today.loc[df_today.focused == 1, "dt"].sum()
    distracted_secs = df_today.loc[df_today.focused == 0, "dt"].sum()
    typing_secs     = df_today.loc[df_today.typing  == 1, "dt"].sum()

    # 5) Taux temporels [%]
    focus_rate      = focus_secs      / total_secs * 100 if total_secs else 0
    distracted_rate = distracted_secs / total_secs * 100 if total_secs else 0
    typing_rate     = typing_secs     / total_secs * 100 if total_secs else 0

    # 6) Formatage des durées pour l’affichage
    focus_fmt      = format_duration(focus_secs)
    distracted_fmt = format_duration(distracted_secs)
    typing_fmt     = format_duration(typing_secs)

    # --- Construction de la sidebar ---
    st.sidebar.header("📊 Report of the day", divider="rainbow")
    st.sidebar.markdown(
        "<p style='text-align: center;'>Here is today's summary of your focus analysis.</p>",
        unsafe_allow_html=True
    )

    # Focus
    st.sidebar.subheader("Focus", divider="blue")
    st.sidebar.markdown(f"""
    <div style='text-align: center; font-size:18px;'>
      🧠 <strong>{focus_rate:.1f}%</strong><br>
      <span style='font-size:16px;'>{focus_fmt} focused over {total_fmt}</span>
    </div>
    """, unsafe_allow_html=True)

    # Distracted
    st.sidebar.subheader("Distracted", divider="orange")
    st.sidebar.markdown(f"""
    <div style='text-align: center; font-size:18px;'>
      👀 <strong>{distracted_rate:.1f}%</strong><br>
      <span style='font-size:16px;'>{distracted_fmt} distracted over {total_fmt}</span>
    </div>
    """, unsafe_allow_html=True)

    # Typing
    st.sidebar.subheader("Typing", divider="violet")
    st.sidebar.markdown(f"""
    <div style='text-align: center; font-size:18px;'>
      ✍️ <strong>{typing_rate:.1f}%</strong><br>
      <span style='font-size:16px;'>{typing_fmt} typed over {total_fmt}</span>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("<br><br>", unsafe_allow_html=True)
    csv_bytes = open(FILENAME_CSV, "rb").read()
    with st.sidebar:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.download_button("⬇️ Download the CSV", data=csv_bytes, file_name=os.path.basename(FILENAME_CSV), mime="text/csv")
        pdf_bytes = make_pdf(df_today)
        with st.sidebar:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.download_button("⬇️ Download the PDF", data=pdf_bytes, file_name=f"summary_{today}.pdf", mime="application/pdf")

        st.sidebar.markdown("<br>", unsafe_allow_html=True)

with st.sidebar.expander("ℹ️ How it works?"):
    st.markdown("""
    - This app analyzes your attention using your webcam.
    - It detects typing, gaze, blinks and focus.
    - Click **Start** to begin, then **Stop** to generate the report.
    """)


# st.write Copyright (c) [2025] GeeksterLab