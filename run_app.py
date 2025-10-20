# run_app.py
from app.app import app

if __name__ == '__main__':
    # debug=True permite que el servidor se reinicie automáticamente cuando haces cambios en el código.
    # ¡Muy útil para desarrollar!
    app.run(debug=True)