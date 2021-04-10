Instrucciones de ejecución

Para ejecutar nuestro código es necesario seguir los siguientes pasos:

  1. Descomprimir la carpeta Codigo que está entregada junto a esta memoria.
  
  2. Descargarse los datos de Bacteria detection with darkfield microscopy de esta página de kaggle:  
  
  https://www.kaggle.com/longnguyen2306/bacteria-detection-with-darkfield-microscopy
  
  3. Descomprimir los datos descargados en el directorio Codigo/data/. Los datos deben de estar accesibles con los paths Codigo/data/images/ y Codigo/data/masks/.
  
  4. Descargar los datos de pre-entrenamiento desde la página:
  
  https://drive.grand-challenge.org/
  
  5. Para poder descargar los datos de DRIVE, primero es necesario registrarse, haciendo click en el apartado ``join''. Luego, podemos pulsar en Download para acceder a los datos subidos en dropbox.
  
  6. Descomprimir los datos descargados en el directorio Codigo/data_pretraining/. Los datos deben de estar accesibles con los paths Codigo/data_pretraining/test/ y Codigo/data_pretraining/train/.
  
  7 Por último, ejecutar el archivo Codigo/proyecto/main.py

La estructura final de directorios debe ser esta:
- BacteriaDetectionMemoria.pdf
- Codigo/
    - data/
        - images/
            - ...
        - masks/
            - ...
    - data_pretraining/
        - test/
            - images/
                - ...
            - masks/
                - ...
        - training/
            - images/
                - ...
            - masks/
                - ...
    - proyecto/
        - saves/
        - loss.py
        - main.py
        - visualization.py

El código main.py está preparado para realizar un pre-entrenamiento y posterior entrenamiento de U-Net y U-Net v2.

Si se quiere ejecutar alguno de los otros experimentos, basta con descomentar la llamada a su respectiva función en el main.