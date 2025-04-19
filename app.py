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
    "¿Estás dispuesto a sacrificar parte de la rentabilidad potencial a cambio de un impacto social o ambiental positivo?",
    "¿Qué opinas sobre el cambio climático?"
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

Luego, genera una pregunta de seguimiento enfocada en la categoría detectada para profundizar en la opinión del inversor.  
"""
prompt_reaccion = PromptTemplate(template=plantilla_reaccion, input_variables=["reaccion"])
cadena_reaccion = LLMChain(llm=llm, prompt=prompt_reaccion)

plantilla_perfil = """
Análisis de respuestas: {analisis}
Genera un perfil detallado del inversor basado en sus respuestas, enfocándote en los pilares ESG (Ambiental, Social y Gobernanza) y su aversión al riesgo. 
Asigna una puntuación de 0 a 100 para cada pilar ESG y para el riesgo, donde 0 indica ninguna preocupación y 100 máxima preocupación o aversión.
Devuelve las 4 puntuaciones en formato: Ambiental: [puntuación], Social: [puntuación], Gobernanza: [puntuación], Riesgo: [puntuación]
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
    st.session_state.mostrar_cuestionario = False
    st.session_state.cuestionario_enviado = False

# Interfaz
st.title("Chatbot de Análisis de Inversor ESG")
st.markdown("""
**Primero interactuarás con un chatbot para evaluar tu perfil ESG.** 
**Al final, completarás un test tradicional de perfilado.**
""")
# Mostrar historial
for mensaje in st.session_state.historial:
    with st.chat_message(mensaje["tipo"]):
        st.write(mensaje["contenido"])

# 1. PREGUNTAS INICIALES
if st.session_state.contador_pregunta < len(preguntas_inversor):
    if not st.session_state.mostrada_pregunta:
        pregunta_actual = preguntas_inversor[st.session_state.contador_pregunta]
        with st.chat_message("bot", avatar="🤖"):
            st.write(pregunta_actual)
        st.session_state.historial.append({"tipo": "bot", "contenido": pregunta_actual})
        st.session_state.mostrada_pregunta = True

    user_input = st.chat_input("Escribe tu respuesta aquí...")

    if user_input:
        st.session_state.historial.append({"tipo": "user", "contenido": user_input})
        st.session_state.respuestas_inversor.append(user_input)
        st.session_state.contador_pregunta += 1
        st.session_state.mostrada_pregunta = False
        st.rerun()
    st.stop()

# 2. NOTICIAS
if st.session_state.contador < len(noticias):
    if not st.session_state.mostrada_noticia:
        noticia = noticias[st.session_state.contador]
        with st.chat_message("bot", avatar="🤖"):
            st.write(f"¿Qué opinas sobre esta noticia? {noticia}")
        st.session_state.historial.append({"tipo": "bot", "contenido": noticia})
        st.session_state.mostrada_noticia = True

    user_input = st.chat_input("Escribe tu respuesta aquí...")

    if user_input:
        st.session_state.historial.append({"tipo": "user", "contenido": user_input})
        st.session_state.reacciones.append(user_input)
        analisis_reaccion = cadena_reaccion.run(reaccion=user_input)
        if len(user_input.split()) < 5:
            with st.chat_message("bot", avatar="🤖"):
                st.write("Podrías ampliar un poco más tu opinión?")
            st.session_state.historial.append({"tipo": "bot", "contenido": "Podrías ampliar un poco más tu opinión?"})
        else:
            st.session_state.contador += 1
            st.session_state.mostrada_noticia = False
            st.rerun()

# 3. PERFIL Y CUESTIONARIO
else:
    if not st.session_state.mostrar_cuestionario:
        analisis_total = "\n".join(st.session_state.reacciones)
        perfil = cadena_perfil.run(analisis=analisis_total)

        try:
            puntuaciones = {
                "Ambiental": int(re.search(r"Ambiental: (\d+)", perfil).group(1)),
                "Social": int(re.search(r"Social: (\d+)", perfil).group(1)),
                "Gobernanza": int(re.search(r"Gobernanza: (\d+)", perfil).group(1)),
                "Riesgo": int(re.search(r"Riesgo: (\d+)", perfil).group(1)),
            }
        except Exception as e:
            st.error(f"No se pudieron extraer las puntuaciones del perfil: {e}")
            st.stop()

        st.session_state.perfil_valores = puntuaciones
        st.session_state.perfil_texto = perfil
        st.session_state.mostrar_cuestionario = True

    # Mostrar perfil y gráfico SIEMPRE
    if "perfil_valores" in st.session_state:
        with st.chat_message("bot", avatar="🤖"):
            st.write(f"**Perfil del inversor:** {st.session_state.perfil_texto}")

        fig, ax = plt.subplots()
        ax.bar(st.session_state.perfil_valores.keys(), st.session_state.perfil_valores.values(), color="skyblue")
        ax.set_ylabel("Puntuación (0-100)")
        ax.set_title("Perfil del Inversor")
        st.pyplot(fig)

    st.header("Cuestionario Final de Perfilado")

    with st.form("formulario_final"):
        objetivo = st.radio("2.1. ¿Cuál es tu objetivo principal al invertir?", ["Preservar el capital (bajo riesgo)", "Obtener rentabilidad moderada", "Maximizar la rentabilidad (alto riesgo)"], index=None)
        horizonte = st.radio("2.2. ¿Cuál es tu horizonte temporal de inversión?", ["Menos de 1 año", "Entre 1 y 5 años", "Más de 5 años"], index=None)

        productos = st.multiselect("3.1. ¿Qué productos financieros conoces o has utilizado?", ["Cuentas de ahorro", "Fondos de inversión", "Acciones", "Bonos", "Derivados (futuros, opciones, CFD)", "Criptomonedas"])
        productos_str = ", ".join(productos) if productos else ""

        volatilidad = st.radio("3.2. ¿Qué significa que una inversión tenga alta volatilidad?", ["Que tiene una rentabilidad garantizada", "Que su valor puede subir o bajar de forma significativa", "Que no se puede vender fácilmente"], index=None)
        largo_plazo = st.radio("3.3. ¿Qué ocurre si mantienes una inversión en renta variable durante un largo periodo?", ["Siempre pierdes dinero", "Se reduce el riesgo en comparación con el corto plazo", "No afecta en nada al riesgo"], index=None)

        frecuencia = st.radio("4.1. ¿Con qué frecuencia realizas inversiones?", ["Nunca", "Ocasionalmente (1 vez al año)", "Regularmente (varias veces al año)"], index=None)
        experiencia = st.radio("4.2. ¿Cuántos años llevas invirtiendo en productos financieros complejos?", ["Ninguno", "Menos de 2 años", "Más de 2 años"], index=None)

        reaccion_20 = st.radio("5.1. ¿Qué harías si tu inversión pierde un 20% en un mes?", ["Vendería todo inmediatamente", "Esperaría a ver si se recupera", "Invertiría más, aprovechando la caída"], index=None)
        combinacion = st.radio("5.2. ¿Cuál de las siguientes combinaciones preferirías?", ["Rentabilidad esperada 2%, riesgo muy bajo", "Rentabilidad esperada 5%, riesgo moderado", "Rentabilidad esperada 10%, riesgo alto"], index=None)

        sostenibilidad = st.radio("6.1. ¿Te interesa que tus inversiones consideren criterios de sostenibilidad?", ["Sí", "No", "No lo sé"], index=None)
        fondo_clima = st.radio("6.2. ¿Preferirías un fondo que invierte en empresas contra el cambio climático aunque la rentabilidad sea menor?", ["Sí", "No"], index=None)
        importancia = st.radio("6.3. ¿Qué importancia das a no financiar sectores controvertidos?", ["Alta", "Media", "Baja"], index=None)

        enviar = st.form_submit_button("Enviar respuestas")

        if enviar:
            try:
                creds_json_str = st.secrets["gcp_service_account"]
                creds_json = json.loads(creds_json_str)
                scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
                creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_json, scope)
                client = gspread.authorize(creds)
                sheet = client.open('BBDD_RESPUESTAS').sheet1

                fila = st.session_state.respuestas_inversor + st.session_state.reacciones + [
                    str(st.session_state.perfil_valores.get("Ambiental", "")),
                    str(st.session_state.perfil_valores.get("Social", "")),
                    str(st.session_state.perfil_valores.get("Gobernanza", "")),
                    str(st.session_state.perfil_valores.get("Riesgo", "")),
                    objetivo or "", horizonte or "", productos_str, volatilidad or "", largo_plazo or "",
                    frecuencia or "", experiencia or "", reaccion_20 or "", combinacion or "",
                    sostenibilidad or "", fondo_clima or "", importancia or ""
                ]

                sheet.append_row(fila)
                st.success("Respuestas enviadas y guardadas exitosamente")
                st.session_state.cuestionario_enviado = True
                st.balloons()
            except Exception as e:
                st.error(f"❌ Error al guardar datos: {str(e)}")

    if st.session_state.cuestionario_enviado:
        st.markdown("### ¡Gracias por completar tu perfil de inversor!")
