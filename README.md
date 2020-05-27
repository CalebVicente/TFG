
						TFG
				ANALISIS BIG DATA DE SUBTITULOS DE TV

--------------------------------------------------------------------------------------------------

VERSION: 27052020
COMMIT: “modules lda and doc2vec working”


Cambios implementados en esta versión

* Implementación de los sistemas doc2vec y lda en modulos que se ejecutarán desde los scripts execute
* Ordenación de la carpeta results
* Creación del k_means de los resultados doc2vec así como la impresión de los mismos en archivos Excel
* Implementación de un archivo config encargado de las rutas de todo el programa
* Función normalize_text y normalize_word dentro de créate_corpus funcionando correctamente tanto las siglas como la unión de nombres compuestos y no lemmatizar nombres de personas

Cambios a implementar en futuras versiones

* ESTÁ DANDO ERRORES CON LOS ARTÍCULOS, CREO QUE CUANDO LA PRIMERA LETRA ESTÁ EN MAYUSCULAS LOS COGE
* Comprobar los key_errors, hay palabras que no las está cogiendo el lda
* Validar el tamaño del vector en doc2vec
* Crear un corpus con el tamaño máximo de los subtítulos para usar en el doc2vec
* Poner todas las rutas del programa en el archivo config
* Ver como clasificar con la similitud que nos proporciona doc2vec y no k_means
* Comprobar si se está creado el corpus para usarlo en el doc2vec


