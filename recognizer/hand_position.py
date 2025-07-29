# recognizer/hand_position.py

def analizar_posicion_manos(data):
    """
    Determina si las manos están por encima o debajo de la cabeza
    con base en los landmarks de pose y manos enviados desde el frontend.
    """

    resultados = []

    # Asegúrate de que existan los landmarks necesarios
    if not data:
        return ["Datos vacíos"]

    pose_landmarks = data.get("pose", [])
    hands_landmarks = data.get("hands", [])

    if len(pose_landmarks) == 0:
        return ["No hay landmarks de pose"]

    # Asumimos que el landmark 0 de pose es la parte superior de la cabeza
    try:
        cabeza_y = pose_landmarks[0]['y']
    except (IndexError, TypeError, KeyError):
        return ["No se pudo determinar la posición de la cabeza"]

    # Procesamos cada mano (izquierda/derecha)
    for mano in hands_landmarks:
        puntos = mano.get("landmarks", [])
        if len(puntos) == 0:
            continue

        # Tomamos la palma (landmark 0) como referencia de la mano
        y_mano = puntos[0]['y']

        # MediaPipe usa coordenadas normalizadas (0-1) y (0,0) es la esquina superior izquierda
        if y_mano < cabeza_y:
            resultados.append("🖐 Mano arriba de la cabeza")
        else:
            resultados.append("✋ Mano abajo de la cabeza")

    if not resultados:
        resultados.append("No se detectaron manos")

    return resultados