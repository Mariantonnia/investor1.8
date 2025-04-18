import streamlit as st
import os
import json
import re
import gspread
import matplotlib.pyplot as plt
from oauth2client.service_account import ServiceAccountCredentials
from dotenv import load_dotenv
from langchain import LLMChain, PromptTemplate
from langchain_groq import ChatGroq

# Cargar variables de entorno
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Configurar LLM
llm = ChatGroq(model="gemma-7b-it", temperature=0)

# Plantilla para generar pregunta de seguimiento
prompt_ampliacion = PromptTemplate(
    input_variables=["texto"],
    template="""
Dado el siguiente texto del usuario:
"{texto}"

Extrae 5 palabras clave y formula una sola pregunta que sirva para ampliar esta opinión desde una perspectiva ESG (ambiental, social, de gobernanza o de riesgo).

Ejemplo:
Texto: "No me parece bien que contaminen tanto"
Palabras clave: [contaminen, medioambiente, regulación, impacto, empresa]
Pregunta: ¿Cómo crees que las regulaciones ambientales deberían aplicarse a las empresas que contaminan?

Tu turno:
"""
)
cadena_ampliacion = LLMChain(llm=llm, prompt=prompt_ampliacion)

# Plantilla para generar el perfil
plantilla_perfil = """
Dado este conjunto de respuestas del usuario:
{analisis}

Genera un perfil del inversor basado en ESG (Ambiental, Social y Gobernanza) y aversión al riesgo.
Asigna puntuaciones de 0 a 100 en el formato:

Ambiental: [puntuación], Social: [puntuación], Gobernanza: [puntuación], Riesgo: [puntuación]
"""
prompt_perfil = PromptTemplate(template=plantilla_perfil, input_variables=["analisis"])
cadena_perfil = LLMChain(llm=llm, prompt=prompt_perfil)

# Noticias de ejemplo
noticias = [
    "Repsol, entre las 50 empresas que más responsabilidad histórica tienen en el calentamiento global",
    "Amancio Ortega crea un fondo de 100 millones de euros para los afectados de la dana",
    "Freshly Cosmetics despide a 52 empleados en Reus, el 18% de la plantilla",
    "Wall Street cae ante la incertidumbre por la guerra comercial y el temor a una recesión",
    "El mercado de criptomonedas se desploma: Bitcoin cae a 80.000 dólares, las altcoins se hunden",
]

# Estado inicial
if "idx" not in st.session_state:
    st.session_state.idx = 0
    st.session_state.historial = []
    st.session_state.reacciones = []
    st.session_state.pendiente_ampliacion = False

# Mostrar historial
for m in st.session_state.historial:
    with st.chat_message(m["tipo"]):
        st.markdown(m["contenido"])

# Flujo principal
if st.session_state.idx < len(noticias):
    if not st.session_state.pendiente_ampliacion:
        noticia = noticias[st.session_state.idx]
        pregunta = f"¿Qué opinas sobre esta noticia? {noticia}"
        st.session_state.historial.append({"tipo": "bot", "contenido": pregunta})
        with st.chat_message("bot"):
            st.markdown(pregunta)
        st.session_state.pendiente_ampliacion = True

    user_input = st.chat_input("Escribe tu respuesta aquí...")
    if user_input:
        st.session_state.historial.append({"tipo": "user", "contenido": user_input})

        palabras_clave = re.findall(r'\w+', user_input)
        if len(palabras_clave) < 5:
            ampliacion = cadena_ampliacion.run(texto=user_input).strip()
            with st.chat_message("bot"):
                st.markdown(ampliacion)
            st.session_state.historial.append({"tipo": "bot", "contenido": ampliacion})
        else:
            st.session_state.reacciones.append(user_input)
            st.session_state.idx += 1
            st.session_state.pendiente_ampliacion = False
        st.rerun()

# Resultado final: Perfil ESG
else:
    if "perfil_generado" not in st.session_state:
        analisis = "\n".join(st.session_state.reacciones)
        perfil = cadena_perfil.run(analisis=analisis)
        st.session_state.perfil_generado = perfil
    else:
        perfil = st.session_state.perfil_generado

    with st.chat_message("bot"):
        st.markdown(f"**Perfil ESG del inversor:**\n\n{perfil}")

    # Graficar
    try:
        puntuaciones = {
            "Ambiental": int(re.search(r"Ambiental: (\d+)", perfil).group(1)),
            "Social": int(re.search(r"Social: (\d+)", perfil).group(1)),
            "Gobernanza": int(re.search(r"Gobernanza: (\d+)", perfil).group(1)),
            "Riesgo": int(re.search(r"Riesgo: (\d+)", perfil).group(1)),
        }

        fig, ax = plt.subplots()
        ax.bar(puntuaciones.keys(), puntuaciones.values(), color="seagreen")
        ax.set_ylabel("Puntuación (0-100)")
        ax.set_title("Perfil del Inversor")
        st.pyplot(fig)

        # Guardar en Google Sheets
        try:
            creds_json_str = st.secrets["gcp_service_account"]
            creds_json = json.loads(creds_json_str)
            scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
            creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_json, scope)
            client = gspread.authorize(creds)
            sheet = client.open('BBDD_RESPUESTAS').sheet1
            fila = st.session_state.reacciones + list(puntuaciones.values())
            sheet.append_row(fila)
            st.success("Datos guardados exitosamente en Google Sheets")
        except Exception as e:
            st.error(f"Error al guardar datos: {str(e)}")

    except Exception as e:
        st.error("No se pudieron extraer puntuaciones. Verifica el formato del perfil.")

# Autofocus input
st.markdown("""
<script>
document.addEventListener('DOMContentLoaded', () => {
    const input = document.querySelector('.stChatInput textarea');
    if(input) input.focus();
});
</script>
""", unsafe_allow_html=True)
