import os
from rembg import remove
from PIL import Image

# Direktori data dan output
data_dir = r'D:\SEMESTER5\PCD\proyek_image_processing_susu\data'
output_dir = r'D:\SEMESTER5\PCD\proyek_image_processing_susu\data_no_bg'

# Membuat folder output jika belum ada
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Loop melalui semua subfolder di dalam data_dir
for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file.endswith('.jpg') or file.endswith('.JPG') or file.endswith('.png'):
            # Buat path gambar input dan output
            image_path = os.path.join(root, file)
            relative_path = os.path.relpath(root, data_dir)
            output_folder = os.path.join(output_dir, relative_path)

            # Buat folder output jika belum ada
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            # Path gambar output
            output_image_path = os.path.join(
                output_folder, file.rsplit('.', 1)[0] + ".png")

            # Memproses dan menghapus background
            input_image = Image.open(image_path)
            output_image = remove(input_image)

            # Menyimpan gambar dengan format PNG (atau JPEG jika diinginkan)
            output_image.save(output_image_path, format="PNG")
            print(f"Processed {image_path}, saved to {output_image_path}")

print("Penghapusan background selesai!")
