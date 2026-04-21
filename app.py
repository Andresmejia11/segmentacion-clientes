import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# ─── Configuración de la página ───────────────────────────────────────────────
st.set_page_config(
    page_title="Segmentación de Clientes B2B",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Estilos ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    .main { background-color: #f8f7f4; }

    .block-container { padding-top: 2rem; padding-bottom: 2rem; }

    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        border: 1px solid #e8e6e1;
        margin-bottom: 0.5rem;
    }
    .metric-card .label {
        font-size: 12px;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 4px;
    }
    .metric-card .value {
        font-size: 28px;
        font-weight: 600;
        color: #1a1a1a;
    }

    .segment-chip {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 500;
        margin: 2px;
    }

    .header-title {
        font-size: 2.2rem;
        font-weight: 600;
        color: #1a1a1a;
        margin-bottom: 0.2rem;
    }
    .header-sub {
        font-size: 1rem;
        color: #666;
        margin-bottom: 2rem;
    }

    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1a1a1a;
        border-left: 3px solid #4f46e5;
        padding-left: 10px;
        margin: 1.5rem 0 1rem 0;
    }

    div[data-testid="stSidebar"] {
        background-color: #1a1a2e;
    }
    div[data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }
    div[data-testid="stSidebar"] .stSelectbox label,
    div[data-testid="stSidebar"] .stFileUploader label {
        color: #a0a0b0 !important;
        font-size: 13px;
    }

    .stButton > button {
        background-color: #4f46e5;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        width: 100%;
    }
    .stButton > button:hover { background-color: #4338ca; }

    .insight-box {
        background: #f0f0ff;
        border-left: 4px solid #4f46e5;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.2rem;
        margin: 0.8rem 0;
        font-size: 14px;
        color: #2d2d5e;
    }

    hr { border-color: #e8e6e1; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ─── Colores por segmento ─────────────────────────────────────────────────────
COLORES_NAT  = ["#6366f1", "#10b981", "#f59e0b"]
COLORES_JUR  = ["#3b82f6", "#8b5cf6", "#ec4899"]
NOMBRES_NAT  = {0: "Recurrentes", 1: "Ocasionales", 2: "Intensivos"}
NOMBRES_JUR  = {0: "Ocasionales", 1: "Recurrentes", 2: "Intensivos"}

VARS = ['TOTAL_VENTAS', 'NUM_COMPRAS', 'NUM_CONSULTAS', 'EMPRESASUNICAS_CONSULT']

# ─── Funciones de carga y procesamiento ───────────────────────────────────────
@st.cache_data(show_spinner=False)
def cargar_datos(clientes_bytes, ventas_bytes, consultas_bytes):
    clientes  = pd.read_csv(clientes_bytes,  sep="|", encoding="latin1")
    ventas    = pd.read_csv(ventas_bytes,    sep="|", encoding="latin1")
    consultas = pd.read_csv(consultas_bytes, sep="|", encoding="latin1")

    ventas_agg = ventas.groupby("ID").agg(
        TOTAL_VENTAS   = ("IMPORTE", "sum"),
        PROMEDIO_VENTA = ("IMPORTE", "mean"),
        NUM_VENTAS     = ("IMPORTE", "count")
    ).reset_index()

    consultas_agg = consultas.groupby("ID").agg(
        NUM_CONSULTAS = ("IDCONSUMO", "count")
    ).reset_index()

    df = (clientes
          .merge(ventas_agg,    on="ID", how="left")
          .merge(consultas_agg, on="ID", how="left"))

    df["NUM_CONSULTAS"]       = df["NUM_CONSULTAS"].fillna(0).astype(int)
    df["TOTAL_VENTAS"]        = df["TOTAL_VENTAS"].fillna(0)
    df["NUM_COMPRAS"]         = pd.to_numeric(df.get("NUM_COMPRAS", 0), errors="coerce").fillna(0)
    df["EMPRESASUNICAS_CONSULT"] = pd.to_numeric(df.get("EMPRESASUNICAS_CONSULT", 0), errors="coerce").fillna(0)

    naturales = ["PERSONA FISICA", "EMPRESARIO"]
    df["TIPO_CLIENTE"] = df["FORMAJURIDICA"].apply(
        lambda x: "NATURAL" if x in naturales else "JURIDICO"
    )

    for col in ["FECHA_REGISTRO", "FECHA_CLIENTE"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")

    drop_cols = [c for c in ["IMPORTE_COMPRAS", "NUM_VENTAS", "CONSUMOSTOTAL"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    return df


@st.cache_data(show_spinner=False)
def segmentar(df, tipo):
    subset = df[df["TIPO_CLIENTE"] == tipo].copy()

    for col in VARS:
        subset[col] = pd.to_numeric(subset[col], errors="coerce").fillna(0)

    # Quitar outliers extremos (p95)
    for col in ["NUM_COMPRAS", "TOTAL_VENTAS"]:
        subset = subset[subset[col] <= subset[col].quantile(0.95)]

    df_log = subset.copy()
    for col in VARS:
        df_log[col] = np.log1p(df_log[col])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_log[VARS].dropna())

    kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42, n_init=20)
    clusters = kmeans.fit_predict(X_scaled)
    subset = subset.iloc[df_log[VARS].dropna().index - df_log[VARS].dropna().index.min()]
    subset = subset.reset_index(drop=True)
    subset["cluster"] = clusters

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    subset["PC1"] = X_pca[:, 0]
    subset["PC2"] = X_pca[:, 1]

    var_exp = pca.explained_variance_ratio_

    return subset, scaler, kmeans, var_exp


@st.cache_data(show_spinner=False)
def entrenar_modelo(_df_seg):
    df_model = _df_seg[_df_seg["cluster"].isin(
        [c for c, cnt in _df_seg["cluster"].value_counts().items() if cnt > 1]
    )].copy()

    X = df_model[VARS].fillna(0)
    y = df_model["cluster"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)
    acc = modelo.score(X_test, y_test)

    importancias = pd.DataFrame({
        "Variable":    VARS,
        "Importancia": modelo.feature_importances_
    }).sort_values("Importancia", ascending=False)

    return modelo, acc, importancias


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📂 Cargar datos")
    st.markdown("Sube los tres archivos `.txt` separados por `|`")

    f_clientes  = st.file_uploader("CLIENTES.txt",  type=["txt", "csv"])
    f_ventas    = st.file_uploader("VENTAS.txt",    type=["txt", "csv"])
    f_consultas = st.file_uploader("CONSULTAS.txt", type=["txt", "csv"])

    st.markdown("---")
    st.markdown("### ⚙️ Opciones")
    tipo_cliente = st.selectbox("Tipo de cliente", ["NATURAL", "JURIDICO"])

    st.markdown("---")
    st.markdown("### ℹ️ Acerca de")
    st.markdown("""
    **TFM · Segmentación B2B**  
    Máster en Ciencia de Datos  
    UOC · 2025
    """)

# ─── Contenido principal ──────────────────────────────────────────────────────
st.markdown('<div class="header-title">Segmentación de Clientes B2B</div>', unsafe_allow_html=True)
st.markdown('<div class="header-sub">Análisis de recurrencia en el sector de información empresarial · Colombia</div>', unsafe_allow_html=True)

if not (f_clientes and f_ventas and f_consultas):
    st.info("👈 Sube los tres archivos en el panel izquierdo para comenzar el análisis.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="label">Paso 1</div>
            <div class="value" style="font-size:18px">📁 Subir datos</div>
            <p style="color:#888;font-size:13px;margin-top:8px">CLIENTES · VENTAS · CONSULTAS</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="label">Paso 2</div>
            <div class="value" style="font-size:18px">🔍 Seleccionar tipo</div>
            <p style="color:#888;font-size:13px;margin-top:8px">Clientes Naturales o Jurídicos</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="label">Paso 3</div>
            <div class="value" style="font-size:18px">📊 Explorar</div>
            <p style="color:#888;font-size:13px;margin-top:8px">Segmentos · Perfiles · Predicción</p>
        </div>""", unsafe_allow_html=True)
    st.stop()

# ─── Procesamiento ────────────────────────────────────────────────────────────
with st.spinner("Cargando y procesando datos..."):
    try:
        df = cargar_datos(f_clientes, f_ventas, f_consultas)
    except Exception as e:
        st.error(f"Error al cargar los datos: {e}")
        st.stop()

with st.spinner("Segmentando clientes..."):
    df_seg, scaler, kmeans_model, var_exp = segmentar(df, tipo_cliente)
    modelo_rf, acc_rf, importancias = entrenar_modelo(df_seg)

nombres = NOMBRES_NAT if tipo_cliente == "NATURAL" else NOMBRES_JUR
colores = COLORES_NAT  if tipo_cliente == "NATURAL" else COLORES_JUR

df_seg["Segmento"] = df_seg["cluster"].map(nombres)

# ─── KPIs ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Resumen general</div>', unsafe_allow_html=True)

total     = len(df_seg)
n_seg     = df_seg["cluster"].nunique()
venta_med = df_seg["TOTAL_VENTAS"].median()
comp_med  = df_seg["NUM_COMPRAS"].median()

k1, k2, k3, k4 = st.columns(4)
for col, label, val in zip(
    [k1, k2, k3, k4],
    ["Clientes analizados", "Segmentos encontrados", "Venta mediana (COP)", "Compras medianas"],
    [f"{total:,}", str(n_seg), f"${venta_med:,.0f}", f"{comp_med:.1f}"]
):
    col.markdown(f"""
    <div class="metric-card">
        <div class="label">{label}</div>
        <div class="value">{val}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📍 Mapa de segmentos",
    "👥 Perfiles",
    "📈 Variables clave",
    "🔮 Predictor"
])

# ── Tab 1: PCA scatter ────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-title">Proyección PCA de los segmentos</div>', unsafe_allow_html=True)

    col_g, col_i = st.columns([3, 1])

    with col_g:
        fig_pca = px.scatter(
            df_seg, x="PC1", y="PC2",
            color="Segmento",
            color_discrete_sequence=colores,
            hover_data={
                "PC1": False, "PC2": False,
                "TOTAL_VENTAS": ":,.0f",
                "NUM_COMPRAS":  True,
                "Segmento":     True
            },
            title=f"Segmentos · Clientes {tipo_cliente.title()}",
            template="simple_white",
            opacity=0.7
        )
        fig_pca.update_traces(marker=dict(size=6))
        fig_pca.update_layout(
            height=440,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            font=dict(family="DM Sans")
        )
        fig_pca.add_annotation(
            text=f"PC1 explica {var_exp[0]*100:.1f}% · PC2 explica {var_exp[1]*100:.1f}% de la varianza",
            xref="paper", yref="paper", x=0, y=-0.12,
            showarrow=False, font=dict(size=11, color="#888")
        )
        st.plotly_chart(fig_pca, use_container_width=True)

    with col_i:
        st.markdown("**Distribución por segmento**")
        counts = df_seg["Segmento"].value_counts()
        for i, (seg, cnt) in enumerate(counts.items()):
            pct = cnt / total * 100
            st.markdown(f"""
            <div style="margin-bottom:12px">
                <span class="segment-chip" style="background:{colores[i]}22;color:{colores[i]};border:1px solid {colores[i]}44">{seg}</span>
                <div style="font-size:22px;font-weight:600;color:#1a1a1a;margin:4px 0">{cnt:,}</div>
                <div style="font-size:12px;color:#888">{pct:.1f}% del total</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="insight-box">La separación visual entre grupos valida la calidad de la segmentación.</div>', unsafe_allow_html=True)

# ── Tab 2: Perfiles ───────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-title">Perfil promedio por segmento</div>', unsafe_allow_html=True)

    perfil = df_seg.groupby("Segmento")[VARS].mean().reset_index()

    labels_es = {
        "TOTAL_VENTAS":            "Ventas totales (COP)",
        "NUM_COMPRAS":             "Nº de compras",
        "NUM_CONSULTAS":           "Nº de consultas",
        "EMPRESASUNICAS_CONSULT":  "Empresas únicas consultadas"
    }

    for var in VARS:
        fig = px.bar(
            perfil, x="Segmento", y=var,
            color="Segmento",
            color_discrete_sequence=colores,
            title=labels_es[var],
            template="simple_white",
            text_auto=".2s"
        )
        fig.update_layout(
            height=300,
            showlegend=False,
            font=dict(family="DM Sans"),
            title_font_size=14,
            yaxis_title="",
            xaxis_title=""
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Tabla de promedios**")
    perfil_display = perfil.copy()
    perfil_display.columns = ["Segmento"] + [labels_es[v] for v in VARS]
    st.dataframe(
        perfil_display.style.format({
            "Ventas totales (COP)":           "{:,.0f}",
            "Nº de compras":                  "{:.1f}",
            "Nº de consultas":                "{:.0f}",
            "Empresas únicas consultadas":    "{:.1f}"
        }),
        use_container_width=True, hide_index=True
    )

# ── Tab 3: Importancia de variables ──────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-title">Importancia de variables (Random Forest)</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns([2, 1])

    with col_a:
        labels_imp = {
            "TOTAL_VENTAS":           "Ventas totales",
            "NUM_COMPRAS":            "Nº compras",
            "NUM_CONSULTAS":          "Nº consultas",
            "EMPRESASUNICAS_CONSULT": "Empresas únicas"
        }
        importancias["Variable_es"] = importancias["Variable"].map(labels_imp)

        fig_imp = px.bar(
            importancias, x="Importancia", y="Variable_es",
            orientation="h",
            color="Importancia",
            color_continuous_scale=["#e0e7ff", "#4f46e5"],
            template="simple_white",
            title="¿Qué variable define mejor el segmento de un cliente?"
        )
        fig_imp.update_layout(
            height=350,
            coloraxis_showscale=False,
            yaxis=dict(autorange="reversed"),
            font=dict(family="DM Sans"),
            title_font_size=14
        )
        fig_imp.update_traces(
            text=importancias["Importancia"].apply(lambda x: f"{x:.1%}"),
            textposition="outside"
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    with col_b:
        st.markdown(f"""
        <div class="metric-card" style="margin-top:2rem">
            <div class="label">Precisión del modelo</div>
            <div class="value">{acc_rf:.1%}</div>
        </div>""", unsafe_allow_html=True)

        top_var = importancias.iloc[0]["Variable_es"] if "Variable_es" in importancias.columns else importancias.iloc[0]["Variable"]
        st.markdown(f"""
        <div class="insight-box" style="margin-top:1rem">
            <strong>Variable clave:</strong> {top_var} es el factor más determinante para clasificar un cliente en su segmento.
        </div>
        <div class="insight-box">
            Un cliente frecuente es más valioso que uno que compra mucho una sola vez.
        </div>
        """, unsafe_allow_html=True)

    # Distribución FM
    st.markdown('<div class="section-title">Plano Frecuencia–Monto por segmento</div>', unsafe_allow_html=True)

    fig_fm = px.scatter(
        df_seg, x="NUM_COMPRAS", y="TOTAL_VENTAS",
        color="Segmento",
        color_discrete_sequence=colores,
        opacity=0.6,
        log_y=True,
        template="simple_white",
        title="Relación entre frecuencia de compra y monto total",
        labels={"NUM_COMPRAS": "Nº de compras", "TOTAL_VENTAS": "Ventas totales (log)"}
    )
    fig_fm.update_traces(marker=dict(size=5))
    fig_fm.update_layout(height=380, font=dict(family="DM Sans"))
    st.plotly_chart(fig_fm, use_container_width=True)

# ── Tab 4: Predictor ──────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-title">Predictor de segmento para un cliente nuevo</div>', unsafe_allow_html=True)
    st.markdown("Ingresa los datos de un cliente y el modelo predecirá a qué segmento pertenece.")

    col_p1, col_p2 = st.columns(2)

    with col_p1:
        ventas_new   = st.number_input("Ventas totales (COP)",           min_value=0.0, value=500.0,  step=100.0)
        compras_new  = st.number_input("Número de compras",              min_value=0,   value=3,      step=1)
    with col_p2:
        consult_new  = st.number_input("Número de consultas",            min_value=0,   value=10,     step=1)
        empunic_new  = st.number_input("Empresas únicas consultadas",    min_value=0,   value=5,      step=1)

    if st.button("🔮 Predecir segmento"):
        X_new = pd.DataFrame([[ventas_new, compras_new, consult_new, empunic_new]], columns=VARS)
        pred  = modelo_rf.predict(X_new)[0]
        proba = modelo_rf.predict_proba(X_new)[0]

        seg_nombre = nombres.get(pred, f"Cluster {pred}")
        idx_color  = list(nombres.keys()).index(pred) if pred in nombres else 0
        color_seg  = colores[idx_color % len(colores)]

        st.markdown(f"""
        <div style="background:{color_seg}18;border:2px solid {color_seg};border-radius:12px;padding:1.5rem;margin-top:1rem;text-align:center">
            <div style="font-size:13px;color:{color_seg};text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px">Segmento predicho</div>
            <div style="font-size:2rem;font-weight:700;color:{color_seg}">{seg_nombre}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Probabilidades por segmento:**")
        clases = modelo_rf.classes_
        for cls, prob in sorted(zip(clases, proba), key=lambda x: -x[1]):
            nombre_cls = nombres.get(cls, f"Cluster {cls}")
            idx_c = list(nombres.keys()).index(cls) if cls in nombres else 0
            col_c = colores[idx_c % len(colores)]
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:10px;margin:6px 0">
                <span style="width:110px;font-size:13px;color:#444">{nombre_cls}</span>
                <div style="flex:1;background:#f0f0f0;border-radius:4px;height:10px">
                    <div style="width:{prob*100:.1f}%;background:{col_c};height:10px;border-radius:4px"></div>
                </div>
                <span style="width:45px;text-align:right;font-size:13px;font-weight:600;color:#1a1a1a">{prob:.1%}</span>
            </div>
            """, unsafe_allow_html=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:#aaa;font-size:12px">TFM · Segmentación de Clientes B2B · Máster en Ciencia de Datos · UOC 2025</div>',
    unsafe_allow_html=True
)
