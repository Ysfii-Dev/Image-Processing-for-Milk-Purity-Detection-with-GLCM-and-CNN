**Image Processing for Milk Purity Detection with GLCM and Convolutional Neural Networks**

Repository ini berisi implementasi analisis citra digital untuk mendeteksi tingkat kemurnian susu menggunakan metode **Gray Level Co-occurrence Matrix (GLCM)** dan **Convolutional Neural Networks (CNN)**. Proyek ini bertujuan untuk mengeksplorasi tekstur dan pola visual dalam citra susu murni dan campuran larutan gula, serta memanfaatkan metode klasifikasi berbasis pembelajaran mesin.

### **Fitur Utama:**

1. **Dataset:**

   - Dataset terdiri dari lima kategori berdasarkan tingkat kemurnian susu: 100%, 90%, 80%, 70%, dan 60%.
   - Gambar diolah terlebih dahulu untuk menghapus latar belakang hitam sehingga fokus pada area susu.

2. **Ekstraksi Fitur dengan GLCM:**

   - Menggunakan GLCM untuk menghitung atribut tekstur seperti korelasi, kontras, energi, dan homogenitas.

3. **Ekstraksi Fitur dengan CNN:**

   - CNN digunakan untuk mengekstrak pola mendalam dari gambar susu, menghasilkan fitur representatif untuk analisis lebih lanjut.

4. **Analisis Grafik Hasil GLCM dan CNN:**

   - Visualisasi dan analisis grafik dari fitur-fitur yang dihasilkan oleh GLCM dan CNN untuk memahami pola data.

5. **Klasifikasi Hasil GLCM dengan MLPClassifier:**
   - Fitur tekstur yang diekstraksi oleh GLCM digunakan untuk pelatihan model **MLPClassifier** sebagai langkah klasifikasi.

### **Struktur Repository:**

- **`data/`**: Folder yang berisi gambar susu dengan berbagai tingkat kemurnian.
- **`output/`**: Folder yang berisi hasil pengolahan data gambar dan analisis dataset.
- **`scripts/`**: Kode sumber untuk preprocessing, ekstraksi fitur, analisis data, dan klasifikasi.
- **`README.md`**: Deskripsi proyek.

### **Tujuan Proyek:**

- Mengukur kualitas susu berdasarkan pola tekstur.
- Menggabungkan metode pemrosesan citra tradisional (GLCM) dan pembelajaran mendalam (CNN) untuk mengetaui kualitas susu.
- melihata perbedaan susu murni dengan susu campuran.

### **Persyaratan:**

- Python 3.8 atau lebih tinggi.
- Pustaka: OpenCV, NumPy, TensorFlow/Keras, scikit-learn, Matplotlib, dan Pandas.

### **Cara Penggunaan:**

1. Clone repository ini.
2. Siapkan dataset Anda sesuai dengan struktur folder.
3. Jalankan skrip preprocessing untuk membersihkan dan menyiapkan gambar.
4. Ekstrak fitur menggunakan GLCM dan CNN.
5. Analisis fitur melalui grafik visualisasi.
6. Gunakan **MLPClassifier** untuk klasifikasi berdasarkan fitur GLCM.
7. Evaluasi model menggunakan metrik akurasi, presisi, recall, dan F1-score.
