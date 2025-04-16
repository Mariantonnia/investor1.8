import streamlit as st
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from langchain import LLMChain, PromptTemplate
from langchain_groq import ChatGroq
import os
import re
import json
from dotenv import load_dotenv
import matplotlib.pyplot as plt

load_dotenv()

# Configurar el modelo LLM
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    model="gemma2-9b-it",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Preguntas iniciales
preguntas_inversor = [
    "¿Cuál es tu objetivo principal al invertir?",
    "¿Cuál es tu horizonte temporal de inversión?",
    "¿Tienes experiencia previa invirtiendo en activos de mayor riesgo como acciones, criptomonedas o fondos alternativos?",
]

# Noticias
noticias = [
    "Repsol, entre las 50 empresas que más responsabilidad histórica tienen en el calentamiento global",
    "Amancio Ortega crea un fondo de 100 millones de euros para los afectados de la dana",
    "Freshly Cosmetics despide a 52 empleados en Reus, el 18% de la plantilla",
    "Wall Street y los mercados globales caen ante la incertidumbre por la guerra comercial y el temor a una recesión",
    "El mercado de criptomonedas se desploma: Bitcoin cae a 80.000 dólares, las altcoins se hunden en medio de una frenética liquidación"
]

# Prompts
plantilla_reaccion = """
Reacción del inversor: {reaccion}
Analiza el sentimiento y la preocupación expresada.  
Clasifica la preocupación principal en una de estas categorías:  
- Ambiental  
- Social  
- Gobernanza  
- Riesgo  

Si la respuesta es demasiado breve o poco clara, solicita más detalles de manera específica.  

Genera una pregunta de seguimiento enfocada en la categoría detectada.  
Ejemplos:  
- Ambiental: "¿Cómo crees que esto afecta la sostenibilidad del sector?"  
- Social: "¿Crees que esto puede afectar la percepción pública de la empresa?"  
- Gobernanza: "¿Este evento te hace confiar más o menos en la gestión de la empresa?"  
- Riesgo: "¿Consideras que esto aumenta la incertidumbre en el mercado?" 
"""
prompt_reaccion = PromptTemplate(template=plantilla_reaccion, input_variables=["reaccion"])
cadena_reaccion = LLMChain(llm=llm, prompt=prompt_reaccion)

plantilla_perfil = """
Análisis de respuestas: {analisis}
Genera un perfil del inversor basado en sus respuestas, con puntuaciones ESG (0-100) y aversión al riesgo. 
Formato de salida:
Ambiental: [puntuación], Social: [puntuación], Gobernanza: [puntuación], Riesgo: [puntuación]
"""
prompt_perfil = PromptTemplate(template=plantilla_perfil, input_variables=["analisis"])
cadena_perfil = LLMChain(llm=llm, prompt=prompt_perfil)

# Estado inicial
if "historial" not in st.session_state:
    st.session_state.historial = []
    st.session_state.contador = 0
    st.session_state.reacciones = []
    st.session_state.respuestas_inversor = []
    st.session_state.contador_pregunta = 0
    st.session_state.mostrada_noticia = False
    st.session_state.mostrada_pregunta = False

st.title("Chatbot de Análisis de Sentimiento")

# Mostrar historial
for mensaje in st.session_state.historial:
    with st.chat_message(mensaje["tipo"]):
        st.write(mensaje["contenido"])

# 1. PREGUNTAS INICIALES (con validación de longitud)
if st.session_state.contador_pregunta < len(preguntas_inversor):
    if not st.session_state.mostrada_pregunta:
        pregunta_actual = preguntas_inversor[st.session_state.contador_pregunta]
        with st.chat_message("bot", avatar="🤖"):
            st.write(pregunta_actual)
        st.session_state.historial.append({"tipo": "bot", "contenido": pregunta_actual})
        st.session_state.mostrada_pregunta = True

    user_input = st.chat_input("Escribe tu respuesta aquí...")

    if user_input:
        # Validar longitud mínima (5 palabras)
        if len(user_input.split()) < 5:
            with st.chat_message("bot", avatar="🤖"):
                st.write("⚠️ Por favor, desarrolla tu respuesta con al menos 5 palabras para entender mejor tu perfil.")
            st.session_state.historial.append({"tipo": "bot", "contenido": "⚠️ Amplía tu respuesta."})
        else:
            st.session_state.historial.append({"tipo": "user", "contenido": user_input})
            st.session_state.respuestas_inversor.append(user_input)
            st.session_state.contador_pregunta += 1
            st.session_state.mostrada_pregunta = False
        st.rerun()
    st.stop()

# 2. NOTICIAS (validación idéntica)
if st.session_state.contador < len(noticias):
    if not st.session_state.mostrada_noticia:
        noticia = noticias[st.session_state.contador]
        with st.chat_message("bot", avatar="🤖"):
            st.write(f"📰 Noticia: {noticia}\n\n¿Qué opinas?")
        st.session_state.historial.append({"tipo": "bot", "contenido": f"📰 {noticia}"})
        st.session_state.mostrada_noticia = True

    user_input = st.chat_input("Escribe tu opinión aquí...")

    if user_input:
        if len(user_input.split()) < 5:
            with st.chat_message("bot", avatar="🤖"):
                st.write("⚠️ Por favor, desarrolla tu opinión con al menos 5 palabras.")
            st.session_state.historial.append({"tipo": "bot", "contenido": "⚠️ Amplía tu opinión."})
        else:
            st.session_state.historial.append({"tipo": "user", "contenido": user_input})
            st.session_state.reacciones.append(user_input)
            analisis_reaccion = cadena_reaccion.run(reaccion=user_input)
            st.session_state.contador += 1
            st.session_state.mostrada_noticia = False
        st.rerun()

# 3. PERFIL Y GUARDADO (igual que antes)
else:
    analisis_total = "\n".join(st.session_state.respuestas_inversor + st.session_state.reacciones)
    perfil = cadena_perfil.run(analisis=analisis_total)

    # Extraer puntuaciones y mostrar gráfico
    puntuaciones = {
        "Ambiental": int(re.search(r"Ambiental: (\d+)", perfil).group(1)),
        "Social": int(re.search(r"Social: (\d+)", perfil).group(1)),
        "Gobernanza": int(re.search(r"Gobernanza: (\d+)", perfil).group(1)),
        "Riesgo": int(re.search(r"Riesgo: (\d+)", perfil).group(1)),
    }

    # Gráfico
    fig, ax = plt.subplots()
    ax.bar(puntuaciones.keys(), puntuaciones.values())
    ax.set_ylabel("Puntuación (0-100)")
    ax.set_title("Perfil ESG y Riesgo")
    st.pyplot(fig)

    # Guardar en Google Sheets (requiere credenciales en st.secrets)
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(st.secrets["gcp_service_account"], scope)
        client = gspread.authorize(creds)
        sheet = client.open('BBDD_RESPUESTAS').sheet1
        fila = st.session_state.respuestas_inversor + st.session_state.reacciones + list(puntuaciones.values())
        sheet.append_row(fila)
        st.success("✅ Datos guardados correctamente.")
    except Exception as e:
        st.error(f"❌ Error al guardar: {e}")
