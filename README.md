# ASL Translator - Lengua de Señas en Tiempo Real

Reconocimiento de señas del lenguaje ASL utilizando OpenCV, MediaPipe y modelos locales entrenados con el dataset WLASL.

## Estructura

- `main.ipynb`: pruebas interactivas.
- `app.py`: servidor Flask.
- `capture/`, `preprocess/`, etc.: componentes modulares.
- `tests/`: pruebas unitarias.

## Requisitos

```bash
pip install -r requirements.txt
```

## Ejecutar
```bash
make run  # convierte el notebook a script
python app.py  # ejecuta el servidor Flask
```

## Roadmap
✅ Detección de manos

⬜ Reconocimiento de palabras básicas

⬜ Traducción inversa (texto → seña)

⬜ Soporte multilingüe (LSM, ISL...)