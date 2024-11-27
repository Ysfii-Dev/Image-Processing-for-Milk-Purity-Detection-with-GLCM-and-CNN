import cv2
import os
import numpy as np
import pandas as pd

# Fungsi untuk membaca gambar dan mengekstraksi area susu berbentuk persegi


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

# Fungsi untuk memproses semua gambar dalam folder dan menghasilkan CSV


def process_images_in_folder(folder_path, output_csv):
    results = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            susu_area = extract_susu_area(image)
            mean_R, mean_G, mean_B = extract_rgb_from_susu(susu_area)
            results.append({
                'filename': filename,
                'mean_R': mean_R,
                'mean_G': mean_G,
                'mean_B': mean_B
            })
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f'Hasil telah disimpan ke {output_csv}')

# Fungsi untuk memproses semua folder dalam folder utama


def process_all_folders(main_folder_path, output_folder_path):
    # Buat folder output jika belum ada
    os.makedirs(output_folder_path, exist_ok=True)

    # Loop melalui semua folder dalam folder utama
    for folder_name in os.listdir(main_folder_path):
        folder_path = os.path.join(main_folder_path, folder_name)
        if os.path.isdir(folder_path):
            # Buat nama file CSV output berdasarkan nama folder
            output_csv = f'{folder_name}_rgb_means.csv'
            output_csv_path = os.path.join(output_folder_path, output_csv)

            # Proses gambar dalam folder ini
            process_images_in_folder(folder_path, output_csv_path)


# Path folder utama yang berisi sub-folder gambar dan folder output
main_folder_path = 'D:/SEMESTER5/PCD/proyek_image_processing_susu/data_no_bg'
output_folder_path = 'D:/SEMESTER5/PCD/proyek_image_processing_susu/output/rgb_means'

# Proses semua folder dalam folder utama dan simpan hasil ke CSV
process_all_folders(main_folder_path, output_folder_path)
