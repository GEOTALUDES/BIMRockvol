#Detectron2 Installation
import os
import subprocess
import sys

def run_command(command):
    try:
        subprocess.check_call(command, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Error ejecutando el comando: {e}")
        sys.exit(1)

def install_dependencies():
    print("Instalando dependencias...")
    run_command(f"{sys.executable} -m pip install --upgrade pip")
    run_command(f"{sys.executable} -m pip install torch torchvision")
    run_command(f"{sys.executable} -m pip install opencv-python-headless")
    run_command(f"{sys.executable} -m pip install 'git+https://github.com/facebookresearch/fvcore'")

def install_detectron2():
    print("Iniciando la instalación de Detectron2 en GitHub Codespaces...")

    # Instalar dependencias
    install_dependencies()

    # Clonar el repositorio de Detectron2
    if not os.path.exists("detectron2"):
        print("Clonando el repositorio de Detectron2...")
        run_command("git clone https://github.com/facebookresearch/detectron2.git")
    else:
        print("El directorio detectron2 ya existe. Omitiendo la clonación.")

    # Instalar Detectron2 en modo editable
    print("Instalando Detectron2...")
    run_command(f"{sys.executable} -m pip install -e detectron2")

    print("Instalación de Detectron2 completada.")

if __name__ == "__main__":
    install_detectron2()

    # Verificar la instalación
    try:
        import detectron2
        import cv2
        print(f"Detectron2 se ha instalado correctamente. Versión: {detectron2.__version__}")
        print(f"OpenCV se ha instalado correctamente. Versión: {cv2.__version__}")
    except ImportError as e:
        print(f"Error: No se pudo importar un módulo. La instalación puede haber fallado. {e}")

print("\nGuía para usar Detectron2 en GitHub Codespaces:")
print("1. Asegúrate de que la instalación se haya completado sin errores.")
print("2. Crea un nuevo archivo Python para tu proyecto.")
print("3. Importa Detectron2 y OpenCV en tu script con 'import detectron2' y 'import cv2'.")
print("4. Utiliza las funcionalidades de Detectron2 según tus necesidades.")
print("5. Si necesitas instalar dependencias adicionales, usa 'pip install' en la terminal.")
print("6. Recuerda guardar tus cambios y hacer commit a tu repositorio de GitHub.")
