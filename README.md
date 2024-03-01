# Laporan Proyek Machine Learning - Putu Widyantara Artanta Wibawa

## Project Overview

Dewasa ini, perkembangan dan pertumbuhan teknologi digital semakin pesat, sehingga keberadaan hal-hal terutama yang bersifat digital menjadi semakin banyak. Hal ini menyebabkan peningkatan jumlah permintaan akan rekomendasi terhadap hal-hal yang bersifat digital. Salah satu hal yang saat ini masih diminati adalah buku atau bahan bacaan. Akibat perkembangan teknologi, saat ini terdapat sangat banyak jenis buku dengan bidang yang beragam. Banyaknya jumlah buku yang ada semakin lama pasti akan menyulitkan pembaca untuk memilih bahan bacaan yang tepat sesuai dengan minat dan preferensi mereka. Hal ini akan menjadi tantangan dan dirasa penting untuk diselesaikan.

Dampak dari semakin banyak buku ini juga dapat berakibat negatif bagi penulis buku maupun percetakan, karena akan membuat pembaca atau pembeli menjadi bingung dalam memilih buku yang sesuai dengan keinginan mereka. Hal ini dapat menyebabkan penurunan profit bagi perusahaan percetakan maupun bagi penulis, selain itu juga berdampak negatif bagi pembaca karena semakin menyulitkan mereka dalam mencari buku. Oleh karena itu, diperlukan sebuah sistem rekomendasi yang dapat memberikan rekomendasi pilihan buku kepada pembaca sehingga memudahkan mereka dalam memilih buku yang ingin dibeli atau dibaca. Selain pada sisi bisnis, sistem rekomendasi buku juga dapat diterapkan pada perpustakaan sehingga dapat memberikan rekomendasi buku kepada siswa-siswa khususnya di sekolah [[1]](https://www.jurnal.stmik-yadika.ac.id/index.php/spirit/article/view/52).

Beberapa penelitian terdahulu telah menggunakan beberapa metode dalam membuat sistem rekomendasi, salah satunya adalah content-based filtering dan collaborative filtering [[2]](https://link.springer.com/chapter/10.1007/978-981-15-0184-5_29) [[3]](https://ieeexplore.ieee.org/abstract/document/7684166). Sistem rekomendasi dengan kedua metode tersebut dapat menyelesaikan permasalahan untuk membuat sistem rekomendasi buku.

## Business Understanding

Berdasarkan latar belakang yang telah dipaparkan, kehadiran sistem rekomendasi buku akan memberikan kemudahan bagi pembaca dalam memilih ataupun membeli buku sesuai dengan preferensinya. Adanya sistem ini akan membantu dan memberikan dampak yang baik bagi perusahaan percetakan maupun penulis buku karena dapat mengakselerasi pembaca untuk kembali memilih dan membeli buku-buku yang membuat mereka tertarik.

### Problem Statements
Berdasarkan latar belakang yang telah dipaparkan, adapun rumusan masalah yang dapat ditarik adalah:
- Bagaimana cara melakukan pengolahan data untuk membuat sistem rekomendasi buku?
- Bagaimana mengembangkan sebuah sistem rekomendasi dengan metode content-based filtering dan collaborative filtering?
- Bagaimana cara untuk mendapatkan dan menampilkan hasil rekomendasi buku kepada pembaca?

### Goals

Berdasarkan rumusan masalah diatas, adapun tujuan yang ingin dicapai adalah:
- Untuk mengetahui cara pengolahan data untuk membuat sistem rekomendasi buku
- Untuk mengembangkan sebuah sistem rekomendasi buku dengan metode content-based filtering dan collaborative filtering
- Untuk mengetahui cara mendapatkan dan menampilkan hasil rekomendasi buku kepada pembaca.

## Solution Approach

- Solusi untuk mengembangkan content-based filtering adalah menggunakan TF-IDF Vectorizer untuk merepresentasikan fitur pada buku dan dilanjutkan dengan Cosine Similarity untuk mendapatkan derajat kesamaan dari buku.
- Solusi untuk mengembangkan metode collaborative filtering adalah dengan mengembangkan sebuah recommendation class yang didapatkan dari arsitektur Neural Network menggunakan Keras Model, kemudian daftar pilihan buku yang belum pernah dibaca akan dipredict oleh model.
- Pada collaborative filtering, diperlukan data user, rating, dan buku akan tetapi pada content-based filtering tidak diperlukan data rating.

## Data Understanding

Dataset yang digunakan dalam mengembangkan sistem rekomendasi buku ini adalah [Book Recommendation Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset) yang dapat diunduh pada Kaggle. Dataset ini terdiri dari 3 file utama, yaitu dataset buku, dataset rating, dan dataset user.

### Books

**Tabel 1. Sample Books Dataset**

| ISBN       | Book-Title           | Book-Author          | Year-Of-Publication | Publisher               | Image-URL-S                              | Image-URL-M                              | Image-URL-L                              |
| ---------- | -------------------- | -------------------- | ------------------- | ----------------------- | ---------------------------------------- | ---------------------------------------- | ---------------------------------------- |
| 0195153448 | Classical Mythology  | Mark P. O. Morford   | 2002                | Oxford University Press | http://images.amazon.com/images/P/019... | http://images.amazon.com/images/P/019... | http://images.amazon.com/images/P/019... |
| 0002005018 | Clara Callan         | Richard Bruce Wright | 2001                | HarperFlamingo Canada   | http://images.amazon.com/images/P/000... | http://images.amazon.com/images/P/000... | http://images.amazon.com/images/P/000... |
| 0060973129 | Decision in Normandy | Carlo D'Este         | 1991                | HarperPerennial         | http://images.amazon.com/images/P/006... | http://images.amazon.com/images/P/006... | http://images.amazon.com/images/P/006... |

Fitur-fitur yang ada pada dataset books:
- `ISBN` atau International Standard Book Number adalah angka pemberi kode unik pada buku.
- `Book-Title` adalah judul buku.
- `Book-Author` adalah penulis buku.
- `Year-Of-Publication` adalah tahun terbit buku.
- `Publisher` adalah perusahaan atau percetakan buku.
- `Image-URL-S` adalah link menuju cover buku berukuran small.
- `Image-URL-M` adalah link menuju cover buku berukuran medium.
- `Image-URL-L` adalah link menuju cover buku berukuran large.

### Ratings

**Tabel 2. Sample Ratings Dataset**

| User-ID | ISBN       | Book-Rating |
| ------- | ---------- | ----------- |
| 276725  | 034545104X | 0           |
| 276726  | 0155061224 | 5           |
| 276727  | 0446520802 | 0           |

Fitur-fitur yang ada pada dataset ratings:
- `User-ID` adalah ID dari user.
- `ISBN` adalah ISBN dari dataset books.
- `Book-Rating` adalah rating yang diberikan oleh user dengan `User-ID` kepada buku dengan `ISBN`

### User

**Tabel 3. Sample Users Dataset**

| User-ID | Location                        | Age  |
| ------- | ------------------------------- | ---- |
| 1       | nyc, new york, usa              | NaN  |
| 2       | stockton, california, usa       | 18.0 |
| 3       | moscow, yukon territory, russia | NaN  |

Fitur-fitur yang ada pada dataset users:
- `User-ID` adalah ID dari user.
- `Location` adalah lokasi atau asal dari user.
- `Age` adalah usia dari user.

# Data Analysis & EDA

![](https://i.ibb.co/kXpv6G1/book-published.png)
**Gambar 1. Total Book Published from 1990 - 2024**

Berdasarkan Gambar 1. dapat diketahui bahwa jumlah buku yang diterbitkan setidaknya dari tahun 1990an sampai tahun 2000an awal cenderung mengalami peningkatan.


![](https://i.ibb.co/Y8f1tyK/book-rating.png)
**Gambar 2. Book Rating Distribution**

Berdasarkan Gambar 2. dapat diketahui bahwa mayoritas rating yang diberikan oleh user adalah 0, kemudian disusul dengan rating 8 dan rating 10.

![](https://i.ibb.co/tCPCJjD/user-location.png)
**Gambar 3. Top 10 User Locations**

Berdasarkan Gambar 3. dapat diketahui bahwa mayoritas user berasal dari Amerika, kemudian disusul Spanyol, dan Inggris.


Selain itu, terdapat beberapa informasi lainnya yang didapatkan yaitu:

- Total unique user: 278858
- Total unique book: 271360
- Total unique author: 102022
- Total unique publisher: 16807

## Data Preparation

1. Menghilangkan missing value dan data yang duplikat. Selain itu juga dilakukan penghilangan outlier pada data user khususnya melalui fitur `Age`, hal ini dilakukan untuk meningkatkan akurasi model dan mengurangi bias.
2. Melakukan penggabungan seluruh dataset menjadi sebuah dataset tunggal menggunakan skema inner join untuk memastikan seluruh data tidak memiliki nilai kosong, hal ini dimaksudkan agar mempermudah proses training dan memastikan integritas data.
3. Untuk content based filtering dilakukan pemilihan 20.000 data pertama untuk mengurangi penggunaan memory yang berlebihan karena jumlah data sangat besar akibat matriks yang terbentuk.

## Modeling

### Content-based Filtering
Pada Content-based Filtering, dimulai dengan memilih 20.000 data pertama yang akan digunakan dalam proses ini. Selanjutnya, data tersebut akan dipastikan bersih dari null values dan duplikat. Setelah data dipastikan bersih, selanjutnya akan dibuat sebuah instance dari TFIDF Vectorizer yang nantinya akan melakukan proses fit dan transform pada data tadi. Kemudian, data yang telah di transform akan dilakukan perhitungan cosine similarity untuk mendapatkan cosine similarity matriks.

Jika matriks sudah didapatkan, maka selanjutnya bisa membuat sebuah function untuk membantu mendapatkan rekomendasi buku dengan mengambil sebanyak k buku terbanyak, tidak termasuk buku yang sudah dibaca oleh user. Selanjutnya pilih salah satu judul buku dan kemudian bisa ditampilkan top 5 rekomendasi untuk buku tersebut.

| ISBN       | Book-Title         | Book-Author  |
| ---------- | ------------------ | ------------ |
| 0743436210 | Hearts in Atlantis | Stephen King |

Berikut adalah hasil rekomendasi buku menggunakan content-based filtering untuk buku Hearts in Atlantis:

| Book-Title                                        | Book-Author  |
| ------------------------------------------------- | ------------ |
| Everything's Eventual : 14 Dark Tales             | Stephen King |
| LA Niebla/Skeleton Crew                           | Stephen King |
| The Stand: The Complete &amp; Uncut Edition       | Stephen King |
| The Green Mile: Coffey's Hands (Green Mile Ser... | Stephen King |
| The Girl Who Loved Tom Gordon                     | Stephen King |

### Collaborative Filtering
Pada Collaborative Filtering, setelah fitur dimapping, selanjutkan akan dibuat sebuah class `BookRecommender` yang melakukan inheritance dari Keras Model. Tahapan yang terdapat pada model ini yakni diawali dengan proses embedding terhadap data user dan data buku. Selanjutnya yaitu dilakukan operasi perkalian dot product antara kedua embedding tersebut. Dapat juga dilakukan penambahan bias pada setiap user dan buku. Tahap terakhir yakni penetapan skor kecocokan dalam skala [0,1] sehingga menggunakan fungsi aktivasi sigmoid.

Model yang dibuat kemudian di compile dengan loss function Binary Crossentropy, optimizer Adam, dan metrik evaluasi Root Mean Squared Error (RMSE). Proses pelatihan model dilakukan dengan batch size 1024 dan jumlah epochs 10.

Model kemudian dilatih dan selanjutnya akan dilakukan proses pencarian rekomendasi buku kepada sebuah user secara acak dengan cara mencari buku-buku yang belum pernah dibaca user, kemudian daftar buku-buku tersebut akan diberikan kepada model untuk di predict sehingga menghasilkan daftar rekomendasi buku.

Berikut adalah hasil rekomendasi buku menggunakan collaborative filtering untuk User-ID 236283.

| Book-Title                                        | Book-Author             |
| ------------------------------------------------- | ----------------------- |
| Out of Nowhere                                    | Doris Mortman           |
| Situation Ethics: The New Morality                | Joseph Francis Fletcher |
| Learning to Say No: Establishing Healthy Bound... | Carla Wills-Brandon     |
| Der Untertan: Roman                               | Heinrich Mann           |
| The Invoker (Lawson Vampire Novels)               | Jon F. Merz             |
| Les Voies d'Anubis                                | Tim Powers              |
| Bad Boys on Board                                 | Lori Foster             |
| Lonely Planet New York City: Condensed (Lonely... | Dani Valent             |
| Nature's Green Umbrella: Tropical Rain Forests    | Gail Gibbons            |
| The Anatomy Coloring Book (2nd Edition)           | Wynn Kapit              |

## Evaluation

### Content-based Filtering
Dalam melakukan evaluasi performa hasil rekomendasi menggunakan *content-based filtering*, digunakan metrik *Precision*. Metrik ini dapat mengukur seberapa tepat sistem rekomendasi dalam memberikan rekomendasi yang tepat. Berikut merupakan persamaan dari *Precision*.

$$Precision = \frac{TP}{TP + FP}$$

Pada persamaan di atas, ${TP}$ atau *True Positive* merupakan jumlah item rekomendasi yang relevan dengan preferensi pengguna, sedangkan ${FP}$ atau *False Positive* merupakan jumlah item rekomendasi yang tidak relevan dengan preferensi pengguna.

Berikut merupakan perhitungan yang dilakukan beserta hasilnya.

$$Precision = \frac{5}{5 + 0}$$
$$Precision = 1$$

Dari hasil perhitungan di atas, apabila dikalikan dengan 100%, maka akan menghasilkan nilai *Precision* sebesar 100%. Maka, dapat disimpulkan bahwa sistem rekomendasi menggunakan *content-based filtering* yang dibuat mampu menghasilkan hasil rekomendasi yang sangat relevan dengan preferensi *user*.

### Collaborative Filtering

Dalam melakukan evaluasi performa hasil rekomendasi menggunakan *collaborative filtering*, digunakan metrik *Root Mean Squared Error* (RMSE). Metrik ini dapat mengukur seberapa jauh tingkat kesalahan model *machine learning* dalam membuat prediksi dibandingkan dengan nilai sebenarnya. Pada proyek ini, nilai yang dimaksud untuk diikutkan dalam evaluasi performa model menggunakan RMSE adalah *rating* dari *user*. Berikut merupakan persamaan dari RMSE.

$$RMSE = \sqrt{\frac{1}{n} \Sigma_{i=1}^n({y}-\hat{y})^2}$$

Pada persamaan di atas, $\hat{y}$ merupakan prediksi *rating* yang dibuat model *machine learning*, ${y}$ merupakan *rating* sebenarnya, dan ${n}$ merupakan jumlah data.

![](https://i.ibb.co/K93BRfF/history.png)

#### Gambar 4. Visualisasi Proses Pelatihan Model **RecommenderNet**

Berdasarkan Gambar 4, dapat diketahui bahwa terjadi tren penurunan RMSE di setiap *epoch* baik pada data latih maupun data validasi. Ini menandakan bahwa model dapat dengan baik memelajari data yang dihadapi. Dari visualisasi proses pelatihan ini, diperoleh nilai eror akhir sekitar 0.3 dengan eror pada data validasi sekitar 0.4.



