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

### 1.2. Notebook Steps

1.  **Mount Google Drive**: Connect your Google Drive to the Colab environment.
2.  **Check GPU**: Verify that a GPU is connected and available for training.
3.  **Copy and Unzip Dataset**: The `dataset.zip` is copied from Google Drive to the Colab environment and unzipped.
4.  **Install Packages**: Install `ultralytics`.
5.  **WandB Login** (optional): Log in to Weights & Biases using an API key stored in Colab's user secrets.
6.  **Initialize YOLO**: Initialize the YOLO model.
7.  **Train Model**:
    *   **Parameters to Adjust**:
        *   `name`: Model name (e.g., "best").
        *   `project`: Project name (e.g., "project").
        *   `epochs`: Number of training epochs (e.g., 10).
        *   `batch_size`: Batch size (e.g., 128, powers of two like 16, 32, 64, 128, 512 are common). Use -1 for autobatch.
        *   `imgsz`: Image size (e.g., 224). Train the YOLO model with the image size you plan to use on the Grove Vision AI V2.
        *   `dataset_path`: Path to your `data.yaml` (e.g., `/content/dataset/data.yaml`).
    *   The model will be trained with `yolo11n.pt` as a pre-trained base.
8.  **Zip and Download Model**: The trained model and project folder are zipped and downloaded to your local machine.

## 2. YOLO11n Full Integer Quantization and VELA Conversion for Grove Vision AI V2

This notebook handles the full integer quantization of your trained YOLO11n model and its conversion using the Arm VELA compiler for deployment on the Himax WiseEye2 (WE2) chip. The results is a full_integer_quant_vela.tflite file.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1LcyD4_aV6UfgQtMo7LzTW76xztIQWI1y?usp=sharing)

### 2.1. Prerequisites and Setup

*   **Python 3.10 Environment**: This notebook requires Python 3.10 due to dependencies on the `imp` module, which is deprecated in newer Python versions. The notebook sets up a virtual environment (`env_yolo11`) with Python 3.10.
*   **Dataset Preparation**: The same dataset structure and zipping (`dataset.zip`) as described in Section 1.1 are required for creating a calibration image set.
*   **Google Drive Setup**: Similar to Section 1.1, ensure `/content/drive/MyDrive/yolo` contains `dataset.zip` and your trained YOLO model (e.g., `best.pt`).

### 2.2. Key Parameters to Adjust

Before running the conversion, carefully review and adjust these parameters in the notebook:

*   `imgsz`: Image size (e.g., 224 or 192, default for Grove Vision AI V2, or other sizes). This *must* match the `imgsz` used during model training.
*   `model_name`: Your custom trained YOLO model name (e.g., `"best.pt"`).
*   `nc`: Number of classes (e.g., 4 for 'amel', 'vcra', 'vespsp', 'vvel').
*   `class_names`: Your class names in the format `'name1', 'name2', ...`.
*   `name`: A custom name for the output VELA compiled model folder.

### 2.3. Notebook Steps

1.  **Mount Google Drive**: Connect your Google Drive.
2.  **Copy and Unzip Dataset**: Copies and unzips the dataset for calibration.
3.  **Create Temporary Dataset for Calibration**: A subset of validation images is created for the calibration process, along with a `temp_data.yaml` file.
4.  **Clone Repositories**: Clones the `YOLO11_on_WE2` repository from HimaxWiseEyePlus and a specific fork of Ultralytics (`kris-himax/ultralytics`).
5.  **Setup Python 3.10 Virtual Environment**: Installs `python3.10-dev` and creates a virtual environment `env_yolo11`, then installs `tensorflow-cpu`.
6.  **Install Ultralytics (kris-himax fork)**: Installs the custom Ultralytics fork within the virtual environment.
7.  **Copy Custom Model and Update Conversion Script**:
    *   Your trained `.pt` model is copied to the `YOLO11_on_WE2` directory.
    *   A `convert_tflite.py` script is dynamically created/updated with your `model_name` and `imgsz` for TFLite conversion.
8.  **Convert to TFLite (Full Integer Quantization)**: The model is exported to a fully integer-quantized TFLite model using the `convert_tflite.py` script within the `env_yolo11` virtual environment.
9.  **Install Ethos-U VELA and Download Config**: Installs `ethos-u-vela` and downloads the `himax_vela.ini` configuration file.
10. **VELA Compilation**: The quantized TFLite model is compiled using the VELA compiler with specific accelerator and system configurations, generating the deployable model.
11. **Zip and Download Results**: The VELA compiled model directory is zipped and downloaded.


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

