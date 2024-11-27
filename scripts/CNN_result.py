import cv2
import os
import numpy as np
import pandas as pd
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

# Fungsi untuk membaca gambar dan mengekstraksi area susu


def extract_susu_area(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    susu_area = image[y:y+h, x:x+w]
    return susu_area

# Fungsi untuk menghitung rata-rata nilai RGB dari area susu yang di-crop


def extract_rgb_from_susu(cropped_image):
    B, G, R = cv2.split(cropped_image)
    mean_R = np.mean(R)
    mean_G = np.mean(G)
    mean_B = np.mean(B)
    return mean_R, mean_G, mean_B

# Fungsi untuk menggunakan CNN (VGG16) sebagai feature extractor dan menghitung fitur tambahan


def calculate_cnn_features(image_channel):
    # Muat model VGG16 (menggunakan weights 'imagenet' dan tanpa layer fully connected)
    model = VGG16(weights='imagenet', include_top=False)

    # Ubah ukuran gambar channel ke 224x224 (ukuran input untuk VGG16)
    image_resized = cv2.resize(image_channel, (224, 224))

    # Tambahkan dimensi untuk saluran warna agar sesuai dengan input model (1, 224, 224, 3)
    image_rgb = cv2.merge([image_resized, image_resized, image_resized])
    image_rgb = np.expand_dims(image_rgb, axis=0)

    # Preprocessing gambar untuk model VGG16
    image_rgb = preprocess_input(image_rgb)

    # Ekstraksi fitur menggunakan CNN (VGG16)
    feature_maps = model.predict(image_rgb)

    # Perhitungan statistik fitur dari feature maps
    mean_activation = np.mean(feature_maps)
    std_activation = np.std(feature_maps)

    # Approximate contrast
    contrast = std_activation / (mean_activation + 1e-7)

    # Approximate dissimilarity
    dissimilarity = np.abs(mean_activation - np.min(feature_maps))

    # Approximate homogeneity
    homogeneity = 1.0 / (1.0 + std_activation)

    # Normalisasi untuk menghitung energi
    feature_maps_normalized = feature_maps / np.max(feature_maps)
    energy = np.sum(feature_maps_normalized ** 2) / \
        feature_maps_normalized.size

    # Cek apakah ada cukup elemen untuk menghitung korelasi
    feature_vector = feature_maps.flatten()
    if len(feature_vector) >= 2:
        # Menghitung korelasi antar dua feature maps pertama
        feature_map1 = feature_maps[0, :, :, 0].flatten()
        feature_map2 = feature_maps[0, :, :, 1].flatten()
        correlation_matrix = np.corrcoef(feature_map1, feature_map2)
        correlation = correlation_matrix[0, 1]
    else:
        correlation = 0

    # Calculate entropy
    entropy = -np.sum(feature_maps * np.log2(feature_maps + 1e-7))

    return mean_activation, std_activation, contrast, dissimilarity, homogeneity, energy, correlation, entropy

# Fungsi untuk memproses semua gambar dalam folder dan menghasilkan CSV


def process_images_in_folder(folder_path, output_csv):
    results = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            susu_area = extract_susu_area(image)

            # Hitung fitur RGB
            mean_R, mean_G, mean_B = extract_rgb_from_susu(susu_area)

            # Ekstrak channel warna dan hitung fitur CNN untuk masing-masing channel
            red_channel = susu_area[:, :, 2]
            green_channel = susu_area[:, :, 1]
            blue_channel = susu_area[:, :, 0]

            # Ekstrak fitur CNN dari masing-masing channel
            mean_R_CNN, std_R_CNN, contrast_R, dissimilarity_R, homogeneity_R, energy_R, correlation_R, entropy_R = calculate_cnn_features(
                red_channel)
            mean_G_CNN, std_G_CNN, contrast_G, dissimilarity_G, homogeneity_G, energy_G, correlation_G, entropy_G = calculate_cnn_features(
                green_channel)
            mean_B_CNN, std_B_CNN, contrast_B, dissimilarity_B, homogeneity_B, energy_B, correlation_B, entropy_B = calculate_cnn_features(
                blue_channel)

            # Simpan hasil dalam list
            results.append({
                'filename': filename,
                'mean_R': mean_R,
                'mean_G': mean_G,
                'mean_B': mean_B,
                'mean_R_cnn': mean_R_CNN,
                'mean_G_cnn': mean_G_CNN,
                'mean_B_cnn': mean_B_CNN,
                'contrast_R': contrast_R,
                'dissimilarity_R': dissimilarity_R,
                'homogeneity_R': homogeneity_R,
                'energy_R': energy_R,
                'correlation_R': correlation_R,
                'entropy_R': entropy_R,
                'contrast_G': contrast_G,
                'dissimilarity_G': dissimilarity_G,
                'homogeneity_G': homogeneity_G,
                'energy_G': energy_G,
                'correlation_G': correlation_G,
                'entropy_G': entropy_G,
                'contrast_B': contrast_B,
                'dissimilarity_B': dissimilarity_B,
                'homogeneity_B': homogeneity_B,
                'energy_B': energy_B,
                'correlation_B': correlation_B,
                'entropy_B': entropy_B
            })

    # Simpan hasil ke CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f'Hasil telah disimpan ke {output_csv}')

# Fungsi untuk memproses semua folder dalam folder utama


def process_all_folders(main_folder_path, output_folder_path):
    os.makedirs(output_folder_path, exist_ok=True)
    for folder_name in os.listdir(main_folder_path):
        folder_path = os.path.join(main_folder_path, folder_name)
        if os.path.isdir(folder_path):
            output_csv = f'{folder_name}_CNN_result.csv'
            output_csv_path = os.path.join(output_folder_path, output_csv)
            process_images_in_folder(folder_path, output_csv_path)


# Path folder utama yang berisi sub-folder gambar dan folder output
main_folder_path = 'D:/SEMESTER5/PCD/proyek_final_susu/data'
output_folder_path = 'D:/SEMESTER5/PCD/proyek_final_susu/output/cnn_results'

# Proses semua folder dalam folder utama dan simpan hasil ke CSV
process_all_folders(main_folder_path, output_folder_path)
