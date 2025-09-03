# YOLO11n Training and Deployment for Grove Vision AI V2

This repository contains two Google Colab notebooks, as well as sections for troubleshooting and references:
- [1. YOLO11n Training on Google Colab](#1-yolo11n-training-on-google-colab)
- [2. YOLO11n Full Integer Quantization and VELA Conversion for Grove Vision AI V2](#2-yolo11n-full-integer-quantization-and-vela-conversion-for-grove-vision-ai-v2)
- [3. Troubleshooting](#3-troubleshooting)
- [4. References](#4-references)


## 1. YOLO11n Training on Google Colab

A notebook to train a Ultralytics YOLO11n object detection model with a custom dataset on Google Colab. 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OAsfMPoQl0vLXD8X9PxToWIX_ZnLuEID?usp=sharing)
### 1.1. Setup and Dataset Preparation

1.  **Dataset Structure**: Organize your dataset with the following folder structure:
    ```
    🗂️ dataset
      🗂️ train
        🗂️ images
        🗂️ labels
      🗂️ valid
        🗂️ images
        🗂️ labels
      data.yaml
    ```
    Ensure `data.yaml` is present in the `dataset` folder.

2.  **Zip the Dataset**: Compress the `dataset` folder into a `dataset.zip` file. On macOS, use the following command to exclude hidden files:
    ```bash
    zip -r dataset.zip . -x "*.DS_Store" "__MACOSX/*" ".Trashes/*" ".Spotlight-V100/*" ".TemporaryItems/*"
    ```

3.  **Google Drive Setup**:
    *   Create a folder named `yolo` in your Google Drive's `MyDrive` (i.e., `/content/drive/MyDrive/yolo`).
    *   Copy the `dataset.zip` file into `/content/drive/MyDrive/yolo`.
    *   Your trained YOLO model (e.g., `best.pt`) will also be saved here.

## 2. YOLO11n Full Integer Quantization and VELA Conversion for Grove Vision AI V2

This notebook handles the full integer quantization of your trained YOLO11n model and its conversion using the Arm VELA compiler for deployment on the Himax WiseEye2 (WE2) chip. The results is a full_integer_quant_vela.tflite file.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1LcyD4_aV6UfgQtMo7LzTW76xztIQWI1y?usp=sharing)

*   **Python 3.10 Environment**: This notebook requires Python 3.10 due to dependencies on the `imp` module, which is deprecated in newer Python versions. The notebook sets up a virtual environment (`env_yolo11`) with Python 3.10.
*   **Dataset Preparation**: The same dataset structure and zipping (`dataset.zip`) as described in Section 1.1 are required for creating a calibration image set.

## 3. Troubleshooting
### Fix class names in the Himax AI web toolkit
- If you use the [Himax AI web toolkit](https://github.com/HimaxWiseEyePlus/Seeed_Grove_Vision_AI_Module_V2/releases/download/v1.1/Himax_AI_web_toolkit.zip) you may find that your custom yolo11n model detects the coco classes (person, bicycle, car, etc). This is because the classes are in the code. You find the list of class names in Himax_AI_web_toolkit/assets/index-legacy.51f14f00.js. Search for person in this file and replace them by the classes you trained your model on.

- ## 4. References

### Object detection with your custom yolo11n model
- How to build the environment on your local computer to make the image file and flash it to the Grove Vision AI V2 on macOS, windows or linux can be found
    - in [How to build yolo11n object detection scenario_app and run on WE2?](https://github.com/HimaxWiseEyePlus/Seeed_Grove_Vision_AI_Module_V2/tree/main/EPII_CM55M_APP_S/app/scenario_app/tflm_yolo11_od#how-to-build-yolo11n-object-detection-scenario_app-and-run-on-we2)
    - and also in [YOLO11n on WE2](https://github.com/HimaxWiseEyePlus/YOLO11_on_WE2)
  
### Detailed information
- Detailed information can be found in this github repository [YOLO11n on WE2](https://github.com/HimaxWiseEyePlus/YOLO11_on_WE2)
    - Install the Yolo11 environment at local PC
    - The output int8 vela tflite model which you can open by [netron](https://netron.app/)
    - The original YOLO11_on_WE2_Tutorial.ipynb on Colab

