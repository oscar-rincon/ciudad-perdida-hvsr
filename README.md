# Proyecto ciudad perdida

## Entorno de Python

El notebook `main/01_location/map.ipynb` usa Python y Folium para crear un mapa HTML interactivo.

### Crear el entorno

Instala [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) o Mamba y, desde la carpeta raíz del proyecto, ejecuta:

```bash
conda env create -f environment.yml
conda activate ciudad-perdida
```

Si usas Mamba, reemplaza `conda` por `mamba` en el primer comando.

### Ejecutar el notebook en VS Code

1. Abre `main/01_location/map.ipynb`.
2. Selecciona el kernel `Python (ciudad-perdida)` o el intérprete del entorno `ciudad-perdida`.
3. Ejecuta las celdas del notebook.

El notebook crea `ciudad_perdida_hvsr_map.html` en su directorio de trabajo. Se necesita conexión a Internet para cargar las capas de mapa.


