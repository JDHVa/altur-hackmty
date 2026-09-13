# Centinela Altur — Guiones de presentación en prosa

> Dos guiones. El **Plan 1** (5–10 min) está pensado para que el juez pruebe el sistema mientras hablamos. El **Plan 2** (15–20 min) explica todo el proyecto y el porqué de cada decisión. Ambos están escritos tal cual se dicen; entre corchetes van indicaciones de escena, no se leen.

---

# PLAN 1 — Demo interactiva (5 a 10 minutos)

[Antes de empezar: la consola abierta en `/console/` del backend GPU (Vultr, sin cold start), un celular con dos audios listos —una voz real del equipo y su clon— y el micrófono probado. Un integrante habla, otro maneja la consola, el tercero atiende al juez.]

## Apertura (45 segundos)

Imagine que de pronto una inteligencia artificial clona su voz. Con diez segundos de un audio de WhatsApp es suficiente. Ahora imagine qué podría hacer alguien con esa voz en un banco: reportar una tarjeta, pedir un cambio de teléfono, autorizar una transferencia. Da miedo, ¿verdad? Pues eso pasa todos los días y se hace en minutos. Y el problema de fondo es que el operador que contesta no tiene ninguna forma de saber si del otro lado hay una persona o hay una máquina.

Nosotros somos Centinela Altur y construimos justo eso: un sistema que escucha la llamada y le dice al banco, con una probabilidad, si quien habla es humano o es una voz sintética. Y en vez de contárselo, preferimos que usted lo pruebe.

## Primera prueba: la voz real (1 minuto)

[Se le entrega el celular o se reproduce el audio real frente al micrófono de la consola; se sube el archivo.]

Este audio es la voz real de uno de nosotros, grabada con el celular, sin ningún tratamiento. Lo subimos a la consola y el sistema hace tres cosas en uno o dos segundos: convierte el audio al formato telefónico, lo pasa por los modelos y nos devuelve una probabilidad.

[Aparece el resultado.]

Ahí está: humano, con una confianza muy alta. Fíjese que el sistema nunca había escuchado esta voz. No la reconoce; lo que hace es medir si suena y se comporta como una persona.

## Segunda prueba: el clon (1 minuto)

[Se sube el clon de la misma voz.]

Ahora viene lo interesante. Esta es exactamente la misma persona, pero clonada con un modelo de inteligencia artificial de código abierto, de los que cualquiera puede descargar hoy. Al oído, muchos no notarían la diferencia.

[Aparece el resultado.]

Sintético. Misma voz, misma frase, y el sistema la separa. Esa es la diferencia entre reconocer una voz y reconocer que una voz no existe.

## Qué está pasando por dentro (2 minutos)

Mientras usted prueba con otro audio si quiere [se le ofrece el micrófono para que grabe algo él mismo], le cuento qué hay detrás.

Para desarrollar esto se nos dio un dataset de trescientas cincuenta y tres llamadas reales a un banco, en español de México, entre un cliente y un agente. En algunas de esas llamadas el cliente era una persona; en otras era una inteligencia artificial completa: reconocimiento de voz, un modelo de lenguaje y una voz sintética marcando el mismo número.

Cuando las escuchamos vimos algo muy interesante. Había dos preguntas distintas escondidas ahí. La primera es cómo se comporta la llamada: cuánto tarda en contestar, si interrumpe, si deja silencios, cuántas veces toma la palabra. Un bot tarda siempre lo mismo en responder y nunca se atropella con el agente; una persona duda, se ríe, se adelanta. De eso sacamos cuarenta y dos números por llamada.

La segunda pregunta es cómo suena la voz. Para eso usamos modelos de voz auto-supervisados, como XLS-R de Meta y WavLM de Microsoft, que ya aprendieron de cientos de miles de horas de audio. No los reentrenamos: los dejamos congelados y solo les ponemos una cabeza pequeña encima que aprende a detectar los artefactos que deja una síntesis. Eso es importante, porque así el modelo aprende a ver la síntesis y no a memorizar la voz de nadie.

Con esas dos familias de señales nos hicimos una pregunta más: ¿podríamos definir estadísticamente la probabilidad de que quien está detrás del teléfono sea un humano? Y la respuesta fue un rotundo sí. Un clasificador de gradient boosting toma las cuarenta y dos señales conversacionales más las tres señales de audio y produce esa probabilidad que usted acaba de ver en pantalla.

## Resultados (45 segundos)

Sobre el split oficial del reto acertamos el cien por ciento de las llamadas: cero falsos positivos y cero falsos negativos. Como cien por ciento suena a que memorizamos algo, lo validamos dos veces más: con validación cruzada en cinco pliegues y cinco semillas nos da noventa y nueve punto tres por ciento, y con un conjunto de setenta y un llamadas que el modelo nunca vio nos da noventa y ocho punto seis. Esa última es la cifra honesta para voces nuevas.

Y esto que está probando no es un notebook. Es un endpoint en producción, con el contrato exacto del reto, corriendo en GPU en la nube. Está en Modal, que escala a cero cuando nadie lo usa, y en Vultr, encendido las veinticuatro horas para que hoy no tuviera que esperar ni un segundo.

## Cierre (20 segundos)

Con Centinela Altur logramos algo que nos parece increíble: sin conocer la voz de alguien, podemos saber si es una persona real o una clonación hecha con inteligencia artificial. Y lo puede probar cualquiera, ahora mismo, desde un navegador.

Centinela Altur: detecta la voz que no existe. Gracias.

[Si sobra tiempo: preguntas. Si el juez quiere probar más audios, se le deja la consola.]

---

# PLAN 2 — Presentación completa del proyecto (15 a 20 minutos)

[Formato: presentación con la consola disponible para una demo al final o en medio. Puede repartirse por áreas: uno cuenta el problema y los datos, otro las señales de audio, otro la fusión y el despliegue.]

## 1. El problema y por qué importa (2 minutos)

Buenas. Somos Centinela Altur y vamos a contarles cómo detectamos voces sintéticas en llamadas bancarias, y sobre todo por qué tomamos cada decisión.

Empiezo por el problema. Clonar una voz hoy cuesta minutos y unos segundos de audio. Con esa voz, un modelo de lenguaje y un sistema de texto a voz, cualquiera puede montar un bot que marque a un banco, se identifique como cliente, conteste preguntas de seguridad y no se canse nunca. Los sistemas tradicionales de verificación por voz están hechos para responder "¿es esta persona?", y ese es justo el punto débil: si la voz es un clon, la respuesta es "sí". La pregunta que hay que hacer es otra: "¿hay una persona aquí?".

El reto de Altur en HackMTY plantea exactamente eso. Nos dan el audio de una llamada telefónica, estéreo a ocho kilohercios, con el cliente en un canal y el agente en el otro, y tenemos que decidir si el cliente es humano o sintético. El entregable es un endpoint: se le manda el audio y responde con una bandera y una confianza. Y hay una condición que marcó todo nuestro diseño: el conjunto con el que nos evalúan es oculto y tiene voces que no aparecen en ningún dato de entrenamiento. Es decir, cualquier cosa que memorice hablantes está condenada a fallar.

## 2. Los datos (2 minutos)

Recibimos trescientas cincuenta y tres llamadas reales, en español mexicano, ya separadas en entrenamiento y validación con hablantes disjuntos. Cada llamada viene con su audio, con los turnos de habla de cada canal y con una etiqueta: humano o sintético.

Lo primero que hicimos fue escucharlas. Y ahí nos dimos cuenta de que había dos historias distintas. Una es la dinámica de la conversación: cuánto tarda el cliente en contestar después de que el agente termina, si se traslapan, cuánto silencio hay, cuántos turnos toma. La otra es la textura de la voz: cómo suena a nivel de señal, con o sin las pequeñas irregularidades que tiene una garganta real.

También aprendimos algo importante sobre el canal. Todo el dataset pasa por una línea telefónica: ocho kilohercios, códec de voz, ruido. Si nosotros entrenábamos o probábamos con audio limpio de estudio, estaríamos comparando peras con manzanas. Por eso todo lo que entra al sistema, incluidos los audios que grabamos nosotros mismos y los clones que generamos para probar, se convierte primero al mismo formato telefónico. Esa disciplina nos evitó varios falsos resultados.

## 3. La señal conversacional: cómo se comporta la llamada (3 minutos)

La primera familia de señales es la conversacional. A partir de los turnos de habla de cada canal calculamos cuarenta y dos números por llamada: la latencia con la que el cliente responde, en promedio y en su variación; el tiempo que tarda en hablar por primera vez; cuántas veces se traslapa con el agente; cuánto silencio total hay; la proporción de tiempo que habla cada uno; el número y la duración de los turnos.

¿Por qué funciona? Porque un bot es demasiado consistente. Responde siempre con la misma latencia, la que le toma transcribir, pensar y sintetizar. Nunca interrumpe, porque espera a que el agente termine. Y no deja los silencios incómodos que dejamos las personas cuando buscamos un número de cuenta. Una persona real es ruidosa en ese sentido: duda, se adelanta, se corrige.

Para obtener los turnos usamos un detector de actividad de voz. Probamos webrtcvad, el de Google, y un detector simple por energía. En el sistema desplegado usamos a propósito el de energía, porque es con el que se entrenó el modelo; cambiar el detector movería la distribución de las cuarenta y dos features y el clasificador dejaría de verlas como las aprendió.

Y hay una razón de fondo para empezar por aquí: estas señales no dependen de quién habla. No codifican identidad. Por eso generalizan a voces nuevas, que es justo lo que nos van a pedir. Además se calculan en milisegundos en un procesador normal, así que con solo esta señal ya teníamos un endpoint válido desde el primer día. Todo lo demás vino a sumar sobre esa base.

## 4. La señal acústica: cómo suena la voz (4 minutos)

La segunda familia mira la señal de audio del cliente. Aquí la tentación es entrenar una red grande desde cero, y es una mala idea con trescientas cincuenta y tres llamadas: la red aprende las voces del dataset y se cae con la primera voz nueva. Lo que hicimos fue apoyarnos en modelos auto-supervisados de voz.

Un modelo auto-supervisado es una red que aprendió a representar el habla sin etiquetas, prediciendo partes ocultas del audio, sobre cientos de miles de horas. Usamos dos: XLS-R de Meta, con trescientos millones de parámetros y entrenado en ciento veintiocho idiomas, y WavLM de Microsoft. Los dos los dejamos completamente congelados. No tocamos ni un peso. Solo entrenamos cabezas pequeñas encima. La ventaja es doble: las representaciones ya saben de fonética, prosodia y canal, y como la parte que entrenamos es chica, no puede memorizar hablantes; solo puede aprender lo que distingue una síntesis de una voz real.

Con XLS-R hicimos algo más fino. El modelo tiene veinticinco capas y cada una codifica cosas distintas: las primeras, textura acústica; las últimas, contenido más abstracto. En lugar de quedarnos con la última, tomamos las veinticinco y aprendemos un peso por capa, lo que en la literatura se llama selección de capas sensibles. El modelo decide solas cuáles capas delatan mejor a una voz sintética. Esa es nuestra señal más fuerte: en un motor de texto a voz que nunca vio en entrenamiento separa perfectamente las dos clases.

Con WavLM hacemos lo directo: tomamos la media y la desviación de su última capa y le ponemos un clasificador. Es una segunda opinión independiente.

La tercera señal es distinta en espíritu. En vez de un clasificador entrenamos dos modelos generativos, uno que aprende la distribución de las voces humanas y otro la de las sintéticas, usando normalizing flows. Ante un audio nuevo, comparamos qué tan probable es bajo cada modelo. Es una razón de verosimilitud, un enfoque clásico de la detección de spoofing pero con densidades modernas. En la versión rápida del sistema esta señal se apaga, porque cuesta tiempo y las otras dos ya cubren el caso, pero está ahí.

Todas las señales acústicas trabajan sobre ventanas de seis segundos y promedian, así el resultado no depende de si la llamada dura veinte segundos o tres minutos.

## 5. La fusión: convertir señales en una probabilidad (2 minutos)

Ya con cuarenta y dos señales conversacionales y tres acústicas, la pregunta que nos hicimos fue si podíamos definir estadísticamente la probabilidad de que quien está detrás del teléfono sea humano. Y la respuesta fue que sí.

No promediamos a ojo. Entrenamos un clasificador de gradient boosting con histogramas, de scikit-learn, con árboles poco profundos y regularización, sobre las cuarenta y cinco entradas. Él aprende cuánto pesa cada señal y cómo interactúan: por ejemplo, que una latencia sospechosa pesa más cuando el audio también suena limpio de más. Elegimos árboles y no una red porque con datos tabulares y pocos ejemplos los árboles ganan, entrenan en segundos y son estables entre semillas.

La salida es una probabilidad, y eso es lo que devolvemos como confianza. Es lo que permite que el sistema no solo diga "sí o no", sino que un operador pueda decidir con criterio: si está muy bajo, atiende normal; si está en zona gris, hace preguntas de seguridad; si está alto, cuelga y escala a fraude.

## 6. Cómo sabemos que funciona (2 minutos)

Después de iterar muchas veces con distintos parámetros, y de descartar cosas que no ayudaban, llegamos a estos números. En el split oficial de entrenamiento a validación, cien por ciento: cero falsos positivos, cero falsos negativos, área bajo la curva de uno. Como eso suena demasiado bien, lo validamos dos veces más. Con validación cruzada estratificada de cinco pliegues y cinco semillas obtenemos noventa y nueve punto veintiséis por ciento, con una desviación de cero punto dos. Y con un holdout de setenta y un llamadas que separamos antes de entrenar y nunca tocamos, noventa y ocho punto seis.

Les cuento también lo que probamos y no sirvió, porque dice mucho de cómo trabajamos. Probamos modelos de detección de deepfake de audio ya entrenados que están en internet: en nuestro dominio, a ocho kilohercios y en español, daban peor que el azar. Probamos aumentar los datos con más códecs telefónicos y con más voces humanas externas, y no mejoró; el dataset ya era diverso. La lección fue que el apalancamiento real estaba en el umbral y en la calidad de las señales, no en más datos.

Y lo probamos con nuestras propias voces. Grabamos a los integrantes del equipo, clonamos esas voces con un modelo abierto y le pedimos al sistema que las separara. Separa las reales de sus clones, y ahí sí encontramos los dos casos más difíciles del proyecto: dos clones casi perfectos que engañan a todas las señales. Ese es el siguiente frente.

## 7. Cómo lo servimos (2 minutos)

Todo esto vive en una carpeta autocontenida con su código y sus pesos. Un contenedor de Docker sobre una imagen de PyTorch con CUDA, un servidor FastAPI que expone el endpoint con el contrato exacto del reto y una consola web para la demo. Si el audio llega en otro formato, ffmpeg lo convierte antes de inferir.

Lo desplegamos en tres lugares por razones distintas. En Modal, que nos da una GPU bajo demanda y escala a cero cuando nadie lo usa, así que casi no cuesta. En Vultr, en una GPU encendida las veinticuatro horas, para que en la evaluación y en la demo no haya arranque en frío: cada llamada se procesa en uno o dos segundos. Y la consola pública en un Space de Hugging Face, que es estático y gratuito, apuntando al backend que queramos.

Además, para el caso de uso real construimos una consola de operador en tiempo real: mientras la llamada sucede, un WebSocket manda el audio en ventanas de un segundo y el operador ve subir o bajar la probabilidad, con un semáforo que le dice continuar, verificar o colgar, y todo queda guardado como serie de tiempo para auditoría. Y estamos llevando el mismo detector a un dispositivo físico con una Raspberry Pi, para que se pueda ver y tocar.

## 8. Cierre (1 minuto)

Resumo en una frase lo que hicimos: miramos cómo se comporta la llamada y cómo suena la voz, fusionamos las dos cosas con un modelo que produce una probabilidad honesta, y lo pusimos en producción con el contrato del reto. Cada decisión, congelar los modelos, usar señales conversacionales, respetar el canal telefónico, validar tres veces, tiene la misma razón detrás: que funcione con voces que nunca hemos escuchado, porque ese es el ataque real.

Con Centinela Altur logramos algo que nos sigue pareciendo increíble: sin conocer la voz de alguien, podemos saber si es una persona real o una clonación hecha con inteligencia artificial.

Centinela Altur: detecta la voz que no existe. Muchas gracias. Quedamos para preguntas y, si quieren, para que prueben el sistema con su propia voz.
