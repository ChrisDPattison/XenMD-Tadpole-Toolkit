# XenMD: TadpoleToolKit 🐸

An automated machine learning pipeline for segmenting and measuring the morphology of *Xenopus tropicalis* tadpoles.

### Option 1: Use the Web App
You can use the no-code web dashboard here: **[INSERT YOUR STREAMLIT LINK HERE LATER]**

### Option 2: Run Locally (Batch Processing)
For Python users who want to batch-process thousands of images on their own hardware:

1. Clone this repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Create a folder named `Data_to_Analyze` in the same directory and place your tadpole images (or nested folders of images) inside.
4. Run the script: `python run_predictions.py`
5. Results will be saved in a newly generated `Analysis_Results` folder.