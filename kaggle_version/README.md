# Kaggle Version - Greedy Adversarial Training

This folder contains everything needed to run your IoT Device Identification project on Kaggle without relying on Google Drive or GitHub.

## 📁 Folder Structure

Your project requires a specific folder structure:

- `./data/` : Place all your **preprocessed** JSON or CSV data here before starting.
- `./models/` : Saved model checkpoints will automatically be stored here.
- `./outputs/` : Final metrics, comparison plots, and dictionaries will go here.

## 🚀 How to Enable GPU on Kaggle

Before running the notebook, ensure that hardware acceleration is enabled to minimize training time:

1. Look at the right-side panel in your Kaggle Notebook editor.
2. Go to **Settings**.
3. Under **Accelerator**, select **GPU T4 x2**.
4. Confirm any pop-up requests to turn on the accelerator.

## 💾 How to Upload Preprocessed Data and Source Code

Since Kaggle disables internet connections for certain competitions/verifications, you need to upload both your preprocessed data and the `src/` folder directly into the notebook environment:

1. Look at the right-side panel and click on **Add Data** -> **Upload**.
2. Upload your data archive containing the `train/`, `val/`, and `test/` splits.
3. **Also upload the `src/` folder included in this directory** as a dataset or drag-and-drop it into the `/kaggle/working` directory structure so the notebook can import the module classes.
4. Once uploaded, the datasets will appear in the `../input/` directory on Kaggle. You can either:
   - Use the Kaggle path directly if you prefer.
   - Manually drag or move your files into the `./data/` folder created by the first cell of the notebook, and move the `src/` folder to the root working directory `./src/`.

## 🏃‍♀️ How to Run the Notebook

1. **Step 1: Create Directories & Setup.** Run the very first cell. It will automatically create all the required folders (`data/`, `models/`, `outputs/`).
2. **Step 2: Check GPU.** Run the `!nvidia-smi` cell to confirm the T4 x2 GPU is correctly attached.
3. **Step 3:** Place your data inside the `data/` folder as instructed.
4. **Step 4: Run All.** You can now execute the rest of the cells sequentially from top to bottom. The codebase has been adapted to use local relative paths exclusively.

> **Note:** The notebook execution relies purely on the local environment and won't require Google Colab, Drive mounts, or `git pull` commands.
