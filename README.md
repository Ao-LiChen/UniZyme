# UniZyme

## Introduction
**UniZyme**: A Unified Protein Cleavage Site Predictor Enhanced with Enzyme Active-Site Knowledge.

This repository contains the code and data for **UniZyme**, a tool designed to predict protein cleavage sites by leveraging enzyme active-site knowledge.

![UniZyme Framework](framework.png)

## Data and Model Weights
The datasets and model weights can be found at: [Data and Model Weights](https://zenodo.org/records/14841050). 
> **Update:** You can find the training, validation, and test splits and model weights for the large-scale baselines presented in the paper under data/Large_Scale_Data_Split.

## Dependencies
Follow the steps below to set up the environment for **UniZyme**:

1. **Install Anaconda** or **Miniconda**.
2. Create a new conda environment and install pytorch, pdbfixer:
    ```bash
    conda create -n UniZyme python=3.8
    conda activate UniZyme
    conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.4 -c pytorch -c nvidia
    conda install -c conda-forge pdbfixer
    ```
3. Install the required dependencies using the provided `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

4. To install the **Frustratometer** library, refer to the instructions provided in the repository: [Frustratometer](https://github.com/HanaJaafari/Frustratometer).
    ```bash
    git clone HanaJaafari/Frustratometer
    cd Frustratometer
    pip install -e .
    ```
   

## Training
To train the models, use the following scripts:

- **UniZyme** (based on pre-trained models):
    ```bash
    python code/train_UniZyme.py
    ```
- **ReactZyme-Variant**:
    ```bash
    python code/train_ReactZyme-Variant.py
    ```
- **ClipZyme-Variant** (based on enzyme features extracted from ClipZyme):
    ```bash
    python code/train_ClipZyme-Variant.py
    ```

The output will be generated and saved in the `output_seq` folder as **.csv** files containing the filtered CDR3B sequences.

## Testing the Model
To test the model and obtain prediction results on various benchmarks:
```bash
python code/Test_UniZyme.py
```

## Running Predictions for Specific Input
To predict cleavage sites for a specific enzyme-substrate pair, use the following command:
```bash
cd code
python predict.py \
  --enzyme_pdb Enzyme.pdb \
  --substrate_pdb Substrate.pdb \
  --enzyme_seq "MKWLLLLSLVVLSECLVKVPLVRKKSLRQNLIKNGKLKDFLKTHKHNPASKYFPEAAALIGDEPLENYLDTEYFGTIGIGTPAQDFTVIFDTGSSNLWVPSVYCSSLACSDHNQFNPDDSSTFEATSQELSITYGTGSMTGILGYDTVQVGGISDTNQIFGLSETEPGSFLYYAPFDGILGLAYPSISASGATPVFDNLWDQGLVSQDLFSVYLSSNDDSGSVVLLGGIDSSYYTGSLNWVPVSVEGYWQITLDSITMDGETIACSGGCQAIVDTGTSLLTGPTSAIANIQSDIGASENSDGEMVISCSSIDSLPDIVFTINGVQYPLSPSAYILQDDDSCTSGFEGMDVPTSSGELWILGDVFIRQYYTVFDRANNKVGLAPVA" \
  --substrate_seq "MGDVEKGKKIFVQKCAQCHTVEKGGKHKTGPNLHGLFGRKTGQAPGFTYTDANKNKGITWKEETLMEYLENPKKYIPGTKMIFAGIKKKTEREDLIAYLKKATNE" \
  --output predictions.csv
  ```
