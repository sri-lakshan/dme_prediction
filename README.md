# DME Prediction Using MaxViT with DR Grade as Input Feature

This project implements a deep learning model for predicting Diabetic Macular Edema (DME) risk by combining retinal fundus images with diabetic retinopathy (DR) grade as an additional feature. The model uses a pretrained MaxViT (Maximally-Deep Vision Transformer) backbone that is fine-tuned for this multimodal task. The project is organized into multiple modules that mimic a real-world workflow:

- **radiologist.py**: Contains the model definition.
- **dataset.py**: Contains the dataset class for loading images and labels.
- **opthamologist.py**: Contains the training routine.
- **train.py**: Acts as an entry point for training the model.

This modular structure is designed to be intuitive, with names resembling roles in a clinical setting.

## Installation

### 1. Clone the Repository

```bash
git clone <repository_url>
cd project
```

### 2. Install Dependencies

Use pip to install the required packages:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file includes:
- torch
- torchvision
- timm
- pandas
- pillow
- scikit-learn
- matplotlib
- numpy

## Data Preparation

## Messidor-2 Dataset

The Messidor-2 dataset is a collection of retinal fundus images used for research in diabetic retinopathy. You can download the dataset from the following sources:

- **ADCIS Official Page**: [https://www.adcis.net/en/third-party/messidor2/]

- **Kaggle Dataset**: [https://www.kaggle.com/datasets/mariaherrerot/messidor2preprocess]

Please ensure you comply with the dataset's usage terms and conditions as specified by the providers.

1. **Images**: Place your retinal fundus images in the designated folder (e.g., `data/images/`).

2. **CSV Files**: You can use `traintestsplit.py` under utils/ in order to split the data into training and validation sets. Create two CSV files named `train_data.csv` and `val_data.csv` in the `data/` directory. Each file should include the following columns:
   - `id_code`: Relative path to the image file.
   - `diagnosis`: The DR grade, represented as an integer (0, 1, 2, 3, or 4).
   - `adjudicated_dme`: The DME label (0 for No DME, 1 for DME).

## Training the Model

To train the model, run the training entry point:

```bash
python src/train.py
```

This process will:
- Load the training data using the custom dataset.
- Build and train the model defined in the project.
- Save the trained model weights for later use.

## Implementation Details

- **Model (radiologist.py)**  
  The model uses a pretrained MaxViT backbone (from the `timm` library) with its classifier head removed. An embedding layer converts the DR grade into a dense vector, which is then concatenated with the image features. Fully connected layers (fusion head) process this fused representation to predict DME.

- **Training Routine (opthamologist.py)**  
  The training routine loads the dataset, applies appropriate image transformations, and trains the model using CrossEntropyLoss and the AdamW optimizer. The training loop iterates over the dataset for a specified number of epochs and saves the final model weights.

## Execution

1. **Train the Model**  
   Execute the training script with:

   ```bash
   python src/train.py
   ```

2. **Evaluation**  
   An evaluation script can be created (or extended from existing code) to load the saved model weights and evaluate model performance on validation data.

## References

- **Clinically-Inspired Multi-Agent Transformers for Disease Trajectory Forecasting From Multimodal Data**
    Nguyen HH, Blaschko MB, Saarakkala S, Tiulpin A.  IEEE Trans Med Imaging. 2024 Jan;43(1):529-541. doi: 10.1109/TMI.2023.3312524. Epub 2024 Jan 2. PMID: 37672368; PMCID: PMC10880139.
    [https://ieeexplore.ieee.org/document/10242080]

- **MaxViT: Multi-Axis Vision Transformer**  
    Ali Hassani et al. (2022)  
    [https://arxiv.org/abs/2204.01697]

- **Diabetic Retinopathy Detection**  
    Gulshan, V., et al. (2016)  
    [https://jamanetwork.com/journals/jama/fullarticle/2588763]

- **Decencière et al..Feedback on a publicly distributed database: the Messidor database.**
    Image Analysis & Stereology, v. 33, n. 3, p. 231-234, aug. 2014. ISSN 1854-5165.
    Available at: [http://www.ias-iss.org/ojs/IAS/article/view/1155] or
    [http://dx.doi.org/10.5566/ias.1155.]

- **Automated analysis of retinal images for detection of referable diabetic retinopathy.**
    M. D. Abràmoff, J. C. Folk, D. P. Han, J. D. Walker, D. F. Williams, S. R. Russell, P. Massin, B. Cochener, P. Gain, L. Tang, M. Lamard, D. C. Moga, G. Quellec, and M. Niemeijer
    JAMA Ophthalmol, vol. 131, no. 3, Mar. 2013, p. 351–357.
    Available at: [https://doi.org/10.1001/jamaophthalmol.2013.1743.]

## Additional Notes

This project simulates a real-world workflow:
- The **Radiologist** (model module) analyzes the images.
- The **Ophthalmologist** (training routine) is responsible for training the model.
- The custom dataset class handles data loading and preprocessing.
- Follow the instructions above to set up your environment and run the program on your machine.
