import os
import sys

def add_to_path(new_path: str):
    module_path = os.path.abspath(os.path.join(new_path))
    if module_path not in sys.path:
        sys.path.append(module_path)

is_colab = False

if is_colab:
    !git clone https://github.com/pdkary/Karys.git
    !cd Karys && git fetch && git pull
    !cd Karys && pip install -r requirements.txt --quiet
    add_to_path('Karys/')
    from google.colab import drive
    drive.mount('/content/drive')
    !cd Karys && pip install -r requirements.txt --quiet
else:
    add_to_path('../../')
    !cd ../../ && pip install -r requirements.txt --quiet