## Integrantes del grupo:

Castaño Jorge

Clavijo Jhonathan

---

# Información para la implementación de la herramienta:

- Clonar el repositorio https://github.com/jhonclavijotro/sum_longT5_base_w_streamlit.git
- Otorgar permisos de ejecución al archivo iniciar.sh
- La aplicación se abrirá en el explorador de internet por defecto

---
## Resumen

El siguiente documento, tiene como objetivo explicar la implementación de un transformer LongT5 propuesto en el año 2022 como una modificación del modelo base T5 de google, en una aplicación desarrollada con la librería de python Streamlit que permita realizar resumenes de documentos PDF y de texto plano, algo que en la actualidad es muy frecuente dentro de lo que es análisis de texto. Como resultado de la implementación, se tiene una herramienta que corre en local con una tarjeta NVIDIA GeForce RTX 3050 Laptop y que permite resumir un documento de no más de 2 páginas y descargar el archivo resultante como un archivo .txt.

## Introducción

El proyecto se basa en la implementación de uno de los checkpoint preentrenados, resultado del artículo LongT5: Efficient Text-To-Text Transformer for Long Sequences. Este modelo se desarrolla como una posible solución a la limitante que tiene de por sí el modelo original T5, el cual, debido a su mecanismo de atención en el enconder (Self-attention) tiene una limitante en cuanto al tamaño de la entrada, ya que la matriz de atención crece de manera lineal con respecto al tamaño de la entrada. Este documento propone dos opciones (atención local y atención global transitoria), las cuales permiten escalar el modelo en cuanto a la entrada sin sacrificar el rendimiento del modelo original y sin agregar un número excesivo de parámetros nuevos.

## Marco teórico

### Transfer Learning en NLP

En el corazón de los modelos modernos de Procesamiento de Lenguaje Natural (NLP) se encuentra el Transfer Learning o aprendizaje por transferencia. Esta técnica permite tomar un modelo pre-entrenado en una tarea con una enorme cantidad de datos y adaptarlo a una tarea específica con muchos menos recursos.

Mientras que en visión computacional este pre-entrenamiento suele ser supervisado (por ejemplo, con ImageNet), en NLP el cambio de paradigma vino con el pre-entrenamiento no supervisado. Modelos como BERT, GPT y T5 se entrenan con enormes corpus de texto sin etiquetar, como Wikipedia o repositorios web, para aprender las sutilezas del lenguaje (sintaxis, semántica, etc.) antes de ser ajustados (fine-tuned) para tareas específicas.

### T5: Unificando Tareas como “Text-to-Text”

El modelo T5 (Text-to-Text Transfer Transformer), presentado por Google en 2019, llevó esta unificación al extremo. Su filosofía es simple pero poderosa: todas las tareas de NLP se tratan como un problema de “texto a texto”.

- Entrada: Un texto que incluye un prefijo que especifica la tarea (ej. resume: ”, ”translate English to Spanish: ”).

- Salida: El texto resultante de dicha tarea.

Esta unificación permitió usar una misma arquitectura, los mismos hiperparámetros y el mismo procedimiento de pre-entrenamiento para una infinidad de tareas.

### Tokenización

La tokenización que utiliza el modelo base T5 es SentencePiece, esta crea un vocabulario de unidades de sub palabras, dividiendo el texto en tokens de subpalabras.

Por ejemplo, ”translate English to French: How are you?” será tokenizado en: [ ’translate’, ’English’, ’_to’, ’French’, ’:’, ’_How’, ’_are’, ’_you’, ’?’ ]

Luego, estos tokens se convierten a IDs y cada token es mapeado a un único ID del vocabulario

Posteriormente, cada uno de estos IDs es convertido a un vector denso (embeddings) y se le agrega un Positional embeddings para conservar la posición de cada token en la secuencia.

### Arquitectura del Transformer Base (Encoder-Decoder)

Para entender LongT5, primero debemos recordar la arquitectura del transformer T5, que sigue la estructura clásica de encoder-decoder:

### Encoder

Su función es leer y comprender la secuencia de entrada. Está compuesto por un stack de bloques, donde cada uno contiene: • Capa de Self-Attention: Calcula las relaciones entre todos los tokens de la entrada utilizando las matrices Q (Queries), K (Keys) y V (Values). En el encoder, esta atención es bidireccional. • Capa Feed-Forward (FFN): Una red neuronal simple que procesa la salida de la atención. • Conexiones Residuales y Normalización (LayerNorm): Se aplica una capa de normalización previa a la subcapa (pre-normalization) para mejorar la estabilidad del entrenamiento.

- Conexiones Residuales y Normalización (LayerNorm): Se aplica una capa de normalización previa a la subcapa (pre-normalization) para mejorar la estabilidad del entrenamiento.

### Decoder

Su función es generar la salida token por token, de forma autorregresiva. Cada bloque incluye:

- Capa de Self-Attention Causal (Enmascarada): Impide que el token actual vea los tokens futuros, asegurando que la predicción en el paso t solo dependa de los pasos anteriores.

- Capa de Cross-Attention (Atención Cruzada): El puente entre encoder y decoder. Aquí, las Q provienen del decoder, mientras que las K y V provienen del encoder, permitiendo consultar el texto original durante la generación.

### Codificación Posicional Relativa

El mecanismo de atención es, por naturaleza, invariante a la posición. T5 utiliza codificación posicional relativa en lugar de embeddings fijos. Esto se logra añadiendo un sesgo (bias) aprendido a los logits de atención que depende de la distancia relativa entre la Key y la Query, lo que facilita la generalización a secuencias más largas que las vistas en el entrenamiento.

### Entrenamiento del T5 (C4)

T5 fue pre-entrenado en el Colossal Clean Crawled Corpus (C4), utilizando una variante de Span Corruption (Corrupción de Tramos):

1. Se selecciona aleatoriamente un 15 % de los tokens de entrada.

2. Estos se agrupan en tramos contiguos y se reemplazan por tokens centinela (<extra_id_0>, <extra_id_1>, etc.).

3. El modelo debe generar como salida los tramos eliminados separados por sus respectivos centinelas.

###  LongT5: Escalando a Documentos Largos

La limitación del T5 original es su complejidad cuadrática O(l2) respecto a la longitud de la secuencia l. LongT5 modifica el mecanismo de atención en el encoder mediante dos estrategias:

###  Atención Local (Local Attention)

Restringe el campo de visión a una ventana de tamaño w. Cada token solo atiende a sus w vecinos más cercanos, reduciendo la complejidad a O(l × w).

###  Atención Global Transitoria (Transient Global Attention o TGlobal)

Permite capturar relaciones de largo alcance sin incurrir en costos cuadráticos:

a. La entrada se divide en bloques de tamaño fijo k.

b. Se calcula un token global (vector de resumen) por bloque mediante suma y normalización.

c. En la atención, cada token local puede atender a su vecindario y a estos l/k tokens globales.

d. La complejidad resultante es O(l × (w + l/k)).

Esta innovación permite a LongT5 manejar hasta 16,384 tokens, manteniendo un rendimiento de vanguardia en resúmenes de documentos largos.

## Metodología

Para la implementación de la herramienta con la librería de python Streamlit, se realizan los siguientes pasos:

- Selección del modelo: Se cuenta con una lista desplegable que permite capturar el nombre del modelo deseado y descargarlo a la máquina para luego ejecutarlo. Es importante resaltar que estos modelos son preentrenados.

- Cargue del documento: Luego de realizar el cargue del checkpoint seleccionado, se carga el archivo de texto que se desea resumir y se suministra al modelo con la sintaxis correspondiente para resumen (el token especial ”summarize: ”seguido del texto cargado sin saltos de línea)

- Salida de respuesta: El modelo luego de aplicar las acciones correspondientes al texto suministrado, entrega el resumen y este se imprime en pantalla, para luego permitir al usuario descargarlo como un texto plano.

## Desarrollo e implementación

El modelo fue diseñado en una máquina linux, por esta razón, el repositorio cuenta con un archivo .sh que permite la ejecución del aplicativo una vez se haya clonado el repositorio. Para la implementación de esta aplicación es necesario realizar los siguientes pasos:

- Clonar repositorio: El repositorio se encuentra en la dirección del repositorio en github

- Permisos de ejecución: Una vez clonado el repositorio, es necesario acceder a la carpeta contenedora y otorgar permisos de ejecución al archivo llamado iniciar.sh, este archivo lo que realiza es la creación de un entorno de python con los requerimientos necesarios para la implementación del modelo y lo ejecuta.

- Uso: Luego de haber ejecutado el archivo .sh, se abrirá una página en el navegador por defecto para archivos html y se podrá cargar el texto y realizar los respectivos resumenes.

´´´
@inproceedings{guo2022longt5,
    title = "{L}ong{T}5: {E}fficient Text-To-Text Transformer for Long Sequences",
    author = "Mandy Guo and Joshua Ainslie and David Uthus and Santiago Onta{\~n}{\'o}n and Jianmo Ni and Yun-Hsuan Sung and Yinfei Yang",
    booktitle = "Findings of the Association for Computational Linguistics: NAACL 2022",
    year = "2022",
    url = "https://aclanthology.org/2022.findings-naacl.55",
    pages = "724--736",
}
´´´
´´´
@misc{uthus2023mlongt5,
    title = "{mLongT5}: A Multilingual and Efficient Text-To-Text Transformer for Longer Sequences",
    author = "David Uthus and Santiago Onta{\~n}{\'o}n and Joshua Ainslie and Mandy Guo",
    year = "2023",
    eprint = "2305.11129",
    archivePrefix = "arXiv",
    primaryClass = "cs.CL",
    url = "https://arxiv.org/abs/2305.11129"
}
´´´