
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from logger_config import logger

def create_datasets(
    csv_file_path: str,
    image_dir_path: str,
    current_path: str,
    barcode_column: str,
    seed: int,
    test_size: float,
    snp_column: int
):
    data = pd.read_csv(csv_file_path)
    data['genotype'] = data[barcode_column].apply(lambda x: 'AA' if x[snp_column] == '0' else 'AC' if x[snp_column] == '1' else 'CC')

    image_file = [i.split("_")[0] for i in os.listdir(image_dir_path)]
    image_filenames_df = pd.DataFrame(image_file, columns=['image_file'])
    image_filenames_df['Tissue Sample ID'] = image_filenames_df['image_file']

    merged_data = pd.merge(image_filenames_df, data, left_on='Tissue Sample ID', right_on='Tissue Sample ID')
    print(len(merged_data))

    shuffled_data = merged_data.sample(frac=1, random_state=seed)
    shuffled_data = shuffled_data.drop(columns=['Tissue Sample ID', 'Unnamed: 0', 'Tissue', 'Patient ID'])

    logger.info("Final dataset (first few rows): \n%s", shuffled_data.head())

    shuffled_data.to_csv(os.path.join(current_path, "data_with_genotype.csv"))

    features = shuffled_data[['image_file', 'barcode']]
    labels = shuffled_data['genotype']

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=seed)

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    y_train_onehot = to_categorical(y_train_encoded)
    y_test_onehot = to_categorical(y_test_encoded)

    class_names = ['AA', 'AC', 'CC']
    logger.info("Class names: \n%s", class_names)

    return (X_train, y_train_onehot), (X_test, y_test_onehot), class_names
