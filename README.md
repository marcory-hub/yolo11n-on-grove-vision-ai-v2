# YOLO11n Training and Deployment for Grove Vision AI V2

This repository contains two Google Colab notebooks, as well as sections for troubleshooting and references:
- [1. YOLO11n Training on Google Colab](#1-yolo11n-training-on-google-colab)
- [2. YOLO11n Full Integer Quantization and VELA Conversion for Grove Vision AI V2](#2-yolo11n-full-integer-quantization-and-vela-conversion-for-grove-vision-ai-v2)
- [3. Troubleshooting](#3-troubleshooting)
- [4. References](#4-references)


## 1. YOLO11n Training on Google Colab

A notebook to train a Ultralytics YOLO11n object detection model with a custom dataset on Google Colab. 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1TGsNgTjzIeN_jRtQZf-Y3opKIXQ82djo?usp=sharing)
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

2.  **Zip the Dataset**: Compress the `dataset` folder into a `dataset.zip` file. On macOS, use the following command to exclude metadata files:
    ```bash
    zip -r dataset.zip . -i '*.jpg' '*.json' '*.yaml' '*.txt' '*.data' '*.names' -x '*.DS_Store' -x '*__MACOSX*' -x '._*'
    ```

3.  **Google Drive Setup**:
    *   Copy the `dataset.zip` file into `/content/drive/MyDrive/`.
  
4. Open the colab notebook, select T4-high RAM and run all code blocks.

5. The results of the yolo11n training will be zipped and downloaded.

6. Unzip the file and copy best.pt that is in the folder `weights` to `/content/drive/MyDrive/`

## 2. YOLO11n Full Integer Quantization and VELA Conversion for Grove Vision AI V2

This notebook handles the full integer quantization of your trained YOLO11n model and its conversion using the Arm VELA compiler for deployment on the Himax WiseEye2 (WE2) chip. The results is a full_integer_quant_vela.tflite file.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1rfAL67MIBLsjftxDs_TmneLN9IPntXIq?usp=sharing)

1. **IMPORTANT**: Select runtime 2025.07 

2. The files are zipped and downloaded including the `full_integer_quant_vela.tflite`.

3. Follow the link to [Himax WiseEye Plus](https://github.com/HimaxWiseEyePlus/Seeed_Grove_Vision_AI_Module_V2) to build the firmware and flash the firmware.

4. Optional: Test it with one of the yolo11n models in the repo.
5. Then copy the `full_integer_quant_vela.tflite` to the modelzoo folder `model_zoo/tflm_yolo11_od/` and adjust the flash command with your `tty.usbmodem#####` and `full_integer_quant_vela.tflite` if needed.
```
python xmodem/xmodem_send.py \
  --port=/dev/tty.usbmodem##### \
  --baudrate=921600 \
  --protocol=xmodem \
  --file=we2_image_gen_local/output_case1_sec_wlcsp/output.img \
  --model=/model_zoo/tflm_yolo11_od/full_integer_quant_vela.tflite,0xB7B000,0x00000
``` 
6. Result can be visualized with [Himax AI web toolkit](https://github.com/HimaxWiseEyePlus/Seeed_Grove_Vision_AI_Module_V2/releases/download/v1.1/Himax_AI_web_toolkit.zip)

## 3. Troubleshooting
### Fix class names in the Himax AI web toolkit
- If you use the [Himax AI web toolkit](https://github.com/HimaxWiseEyePlus/Seeed_Grove_Vision_AI_Module_V2/releases/download/v1.1/Himax_AI_web_toolkit.zip) you may find that your custom yolo11n model detects the coco classes (person, bicycle, car, etc). This is because the classes are in the code. You find the list of class names in Himax_AI_web_toolkit/assets/index-legacy.51f14f00.js. Search for person in this file and replace them by the classes you trained your model on.

- ## 4. References

### Object detection with your custom yolo11n model
- How to build the environment on your local computer to make the image file and flash it to the Grove Vision AI V2 on macOS, windows or linux can be found
    - in [How to build yolo11n object detection scenario_app and run on WE2?](https://github.com/HimaxWiseEyePlus/Seeed_Grove_Vision_AI_Module_V2/tree/main/EPII_CM55M_APP_S/app/scenario_app/tflm_yolo11_od#how-to-build-yolo11n-object-detection-scenario_app-and-run-on-we2)
    - and also in [YOLO11n on WE2](https://github.com/HimaxWiseEyePlus/YOLO11_on_WE2)
  
### References
- [YOLO documentation](https://docs.ultralytics.com/) from Ultralytics. 
- Github repository [YOLO11n on WE2](https://github.com/HimaxWiseEyePlus/YOLO11_on_WE2)
- The output int8 vela tflite model can be visualized in [netron](https://netron.app/)


