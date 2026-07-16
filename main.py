"""
main.py
====================================================================
Lanzador (Entrypoint) del Monitor de Dolor XAI-EEG.

Este script facilita el inicio de la aplicación Streamlit sin tener
que recordar la ruta exacta del archivo de interfaz.
Simplemente ejecuta: python main.py
====================================================================
"""
import os
import sys
import subprocess
from pathlib import Path

def main():
    # Detectar la ruta raíz del proyecto dinámicamente
    root_dir = Path(__file__).resolve().parent
    
    # Ruta exacta hacia el orquestador principal de Streamlit
    app_path = root_dir / "src" / "frontend" / "app.py"

    # Verificación de seguridad
    if not app_path.exists():
        print(f"❌ Error crítico: No se encontró el archivo de interfaz en:\n{app_path}")
        print("Asegúrate de no haber borrado o movido 'src/frontend/app.py'")
        sys.exit(1)

    print("🧠 Iniciando el Monitor de Dolor XAI-EEG...")
    print("-------------------------------------------------")
    
    # Lanzar la aplicación Streamlit
    try:
        subprocess.run(["streamlit", "run", str(app_path)])
    except KeyboardInterrupt:
        print("\n🛑 Aplicación detenida por el usuario. ¡Hasta pronto!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Ocurrió un error al intentar iniciar Streamlit: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
