# Laporan Proyek Machine Learning - Hendra

## Domain Proyek
### Lantar Belakang
Sebagai investor saham, baik pemula maupun yang telah berpengalaman, semuanya membutuhkan strategi guna memaksimalkan hasil dari investasi. Dalam hal ini, analisis teknikal saham dapat digunakan untuk menilai kondisi pasar saat ini berdasarkan histori harga di masa lampau sekaligus memberikan gambaran mengenai pergerakan pasar di masa depan.
<div>
    <img src="https://indiaforensic.com/wp-content/uploads/2012/12/Stock-market12.jpg" width="900"/>
</div>
Pasar saham sendiri merupakan tempat dimana para investor terhubung untuk melakukan transaksi jual beli saham perusahaan publik baik melalui bursa maupun di luar dari bursa. Kegiatan di dalam pasar saham sudah diatur di dalam regulasi yang telah dibuat oleh pemerintah. Aturan hukum tentang regulasi pasar saham sudah diatur dalam Undang-undang Nomor 8 Tahun 1995 tentang Pasar Modal. Undang-undang tersebut berisikan aturan dan ketentuan mengenai aktivitas di pasar modal.

Referensi: [Mandiri-Online-Securitas-Trading Tentang Pasar Saham](https://www.most.co.id/belajar-investasi/saham/tentang-pasar-saham)

## Business Understanding
Proyek ini dibangun untuk perusahaan dengan karakteristik bisnis sebagai berikut :

+ Management Investasi yang ingin meningkatkan akurasi dalam Forecasting Harga saham di jangka menengah dan panjang.
+ Perusahaan Sekuritas Saham yang ingin memantau trend setiap saham dengan potensi keuntungan untuk di tampilkan ke sisi Penguna.
+ Trader individual yang ingin membuat strategi Tranding untuk memaksimalkan profit, ataupun Investor individual yang ingin membuat strategi untuk membeli saham di harga yang lebih murah.

### Problem Statement

1. Fitur apa yang paling berpengaruh terhadap harga saham?
2. Bagaimana cara memproses data agar dapat dilatih dengan baik oleh model?
3. Berapa harga saham berdasarkan kondisi teknikal dengan parameter tertentu pada jam perdagangan?

### Goals

1. Mengetahui kombinasi fitur yang berpengaruh pada pergerakan harga saham.
2. Melakukan persiapan data untuk dapat dilatih oleh model.
3. Membuat model Multivariate Time Series Forecasting yang dapat memprediksi harga saham berdasarkan kondisi teknikal dengan parameter tertentu pada jam perdagangan dengan tingkat akurasi tertinggi.

### Solution Statement

1. Menganalisis data dengan melakukan univariate `Augmented Dickey-Fuller` analysis dan multivariate  `Augmented Dickey-Fuller` analysis yang di kombinasikan dengan visualisasi Data untuk memahami relasi setiap variable untuk pergerakan harga saham.
2. Memilih fitur yang diperlukan, menyiapkan data dengan Scaling dataset, Train Test Split dan Transform Dataset agar bisa digunakan dalam pelatihan model.
3. Membangun model Multivariate Time Series Forecasting dengan memakai Recurrent neural network serta memamfaatkan LSTM dan Bidirectional LSTM.

## Data Understanding
Untuk dataset saya memakai Dataset dari Yahoo Finance, yaitu dataset history harga saham PT Astra Internasional dari 2001 - 1 Oktober 2022.
[Yahoo Finance Pt Astra Internasional Historical Data](https://finance.yahoo.com/quote/ASII.JK/history?p=ASII.JK).
Dataset diambil dari library yFinance, Berikut informasi pada dataset:

+ Dataset memiliki format CSV (Comma-Seperated Values).
+ Dataset memiliki 7 fitur dengan 5494 sample untuk limit pengambilan data History sampai 1 oktober 2022.
+ Dataset memiliki 1 fitur bertipe Timestamp dan 6 fitur bertipe Float.
+ Dataset tidak memiliki missing value.

### Variable - variable pada dataset
Dataset Yahoo Finance ASII.JK History memiliki variable sebagai berikut:
- Date: tanggal dan waktu  pada hari perdagangan
- Open: harga pembukaan saham  pada hari perdagangan
- High: harga tertinggi saham  pada hari perdagangan
- Low: harga terendah saham  pada hari perdagangan
- Close: Harga saham penutupan perdagangan
- Adj Close: Harga penutupan di hitung dari 30 menit penutupan
- Volume: Volume Transaksi Bid/Offer pada hari perdagangan

Time Series terdiri dari tiga komponen sistematis termasuk level, tren, Season, dan komponen non-sistematis yang disebut Noise.
Komponen-komponen ini didefinisikan sebagai berikut:
- Level: Nilai rata-rata dalam seri.
- Trend: Nilai naik atau turun dalam deret.
- Season: Siklus jangka pendek yang berulang dalam seri.
- Noise: Variasi acak dalam seri.

maka untuk memahami data tersebut, diperlukan visualisasi data untuk semua key tersebut dengan X axis mengunakan tanggal perdagangan dan Y axis memakai variable dari Yahoo Finance ASII.JK History dataset
- LinePlot untuk semua Variable Yahoo Finance ASII.JK History dataset:
    <div>
    <img src="https://github.com/hendradra1234/Machine-Learning/blob/master/linePlotAllFeature.png?raw=true" width="800"/>
    </div>
- scatterplot untuk semua Variable Yahoo Finance ASII.JK History dataset:
    <div>
    <img src="https://github.com/hendradra1234/Machine-Learning/blob/master/scatterplotAllFeature.png?raw=true" width="800"/>
    </div>
Kedua Chart tersebut menunjukkan pola hubungan antara volume dan harga saham, di saat volume tinggi maka harga saham akan naik.

### Tes ADF (Augmented Dickey-Fuller)
Tes Dickey-Fuller adalah salah satu tes statistik yang paling populer. Ini dapat digunakan untuk menentukan keberadaan akar satuan dalam deret, dan karenanya membantu kita memahami apakah deret itu stasioner atau tidak. Hipotesis nol dan alternatif dari tes ini adalah:

- Hipotesis Null: Deret memiliki akar satuan (nilai a = 1)

- Hipotesis Alternatif: Deret tidak memiliki akar satuan.

Jika gagal menolak null hypothesis, deret tersebut tidak stasioner. Ini berarti bahwa deret tersebut dapat linier atau stasioner beda. Jika mean dan standar deviasi keduanya adalah garis datar (constant mean dan constant variance), deret tersebut menjadi stasioner.
Test ADF pada Penelitian ini di implementasikan sebagai berikut:
<div>
    <img src="https://github.com/hendradra1234/Machine-Learning/blob/master/MeanStandarDeviationPlot.png?raw=true" width="800"/>
</div>
- Test Stationary spesifik untuk variable [Open, Close dan Volume]

- Open:
    <div>
    <img src="https://github.com/hendradra1234/Machine-Learning/blob/master/StationalityOpen.png?raw=true" width="800"/>
    </div>
- Close
    <div>
    <img src="https://github.com/hendradra1234/Machine-Learning/blob/master/StationalityClose.png?raw=true" width="800"/>
    </div>
- Volum:
    <div>
    <img src="https://github.com/hendradra1234/Machine-Learning/blob/master/StationalityVolume.png?raw=true" width="800"/>
    </div

## Data Preparation
Data Preparation diperlukan untuk menghasilkan dataset yang bisa di gunakan untuk Training model, Proses Data Preparation yaitu sebagai berikut
+ `Feature Selection`
    merupakan metode filtering untuk mengambil variable yang di perlukan untuk training, pada dataset ini Adj Close dan penambahan variable Prediction dengan data dari variable Close.
+ `Scaling Dataset`
    merupakan metode untuk membuat data dalam skala yang sama, hal ini berguna karena Training model akan menjadi lebih optimal jika dataset memiliki skala yang sama, proses Scaled dataset dilakukan dengan Sklearn `MinMaxScaler`.
    Dikarenakan model yang dilatih pada Scaled data akan menghasilkan prediksi berbentuk Scaled. Oleh karena itu, ketika membuat prediksi nanti dengan model, maka hasil prediksi tersebut harus di skala kembali ke bentuk awal. Scaler_model akan beradaptasi dengan bentuk data Array 5 dimensi. Namun, prediksi model akan berbentuk Array satu dimensi. Karena scaler memiliki bentuk input yang tetap, maka model tidak bisa begitu saja menggunakannya kembali untuk menghapus skala prediksi model yang di buat. Untuk menghapus skala prediksi nanti, karena itu dibuat scaler tambahan yang berfungsi pada satu kolom fitur `scaler_pred`.
+ `Train Test Split`
   merupakan proses membagi data menjadi data latih dan data uji. Data latih akan digunakan untuk membangun model, sedangkan data uji akan digunakan untuk menguji performa model. Pada proyek ini dataset sebesar 5494 dibagi menjadi 4346 (80%) untuk Train data dan 1098 (20%) untuk test Data.
+ `Transform Dataset` merupakan proses membuat dataset yang sudah di Scaled menjadi bentuk mini-batch yang bisa di gunakan untuk melatih model.
    Karena model regresi multivariat di modelkan berdasarkan struktur data tiga dimensi. Dimensi pertama adalah urutan, dimensi kedua adalah langkah waktu (mini-batch), dan dimensi ketiga adalah fitur.
    Dataset `train` dan `test` yang telah di Scaled akan di Transform menjadi:
    + `x_train` berisi 4346 urutan input, masing-masing dengan 50 batch dan 5 fitur.
    + `y_train` berisi 4346 nilai target.
    + `x_test` berisi 1098 urutan input, masing-masing dengan 50 batch dan 5 fitur.
    + `y_test` berisi 1098 nilai target.
+ Menentukan n_neuron dan nilai shape
Penentuan jumlah neuron dan Shape ini dilakukan agar jumlah neuron pada Layer pertama menjadi sama dengan input data yang terdiri dari 50 steps dan 5 fitur. Untuk model yang di gunakan, nilai neuron dan shape ditentukan menjadi:
    + n_neuron: 250.
    + steps: 50.
    + feature: 5.

## Modeling

### Algoritma
Penelitian ini dilakukan dengan Tensorflow dan mengunakan algoritma Recurrent Neural Networks (RNN), model sendiri memakai metode Regression Multivarian, yang di mana terdapat 5 parameter yang dipakai untuk Input Shape. model dituliskan dengan mengunakan Layer berikut:
+ LSTM
    Pengunaan LSTM dipilih karena LSTM dapat menyajikan informasi dari riwayat penyimpanan yang telah disimpan cukup lama. LSTM juga mampu untuk mengklasifikasikan dan menghapus informasi yang telah usang. Kemampuan itu membuat LSTM cocok untuk implementasi Time Series Forecasting.
    + `units` merupakan nilai yang digunakan untuk menentukan ruang dimensi output.
    + `return_sequences` untuk menentukan akan return output sequence atau tidak, dengan nilai default False.
    + `inputs` merupakan 3D tensor dengan shape [batch, timesteps, feature].
+ BIDIRECTIONAL LSTM
    Bidirectional LSTM merupakan LSTM yang bisa mengambil data secara dua arah tampa mengabaikan konteks dari data yang di baca sehingga model bisa memberikan prediksi yang lebih akurat.
    + `units` merupakan nilai yang digunakan untuk menentukan ruang dimensi output.
    + `return_sequences` menentukan akan return output sequence atau tidak, dengan nilai default False.
+ Dense
    Dense Layer merupakan lapisan Neural Network yang terhubung secara teratur. Ini adalah Layer yang paling umum dan sering digunakan. Dense Layer mengambil data semua output dari Layer sebelumnya ke semua neuronnya, setiap neuron memberikan satu output ke lapisan berikutnya.
    + `units` merupakan nilai yang digunakan untuk menentukan ruang dimensi output.

### Optimizer
Pada Penelitian ini, Model mengunakan Optimizer sebagai berikut:
+ `Adam`
 adalah algoritma optimisasi yang dapat digunakan sebagai ganti dari prosedur classical         stochastic gradient descent untuk memperbarui bobot secara iteratif yang didasarkan pada data training. parameter yang digunakan untuk penelitian ini meliputi:
    + `Alpha` juga dikenal sebagai learning rate atau step size. Bobot yang proporsional memiliki nilai 0.001. Nilai yang lebih besar (mis. 0.3) menghasilkan pembelajaran awal yang lebih cepat sebelum tarif diperbarui. Nilai yang lebih kecil (mis. 1.0E-5) memperlambat pembelajaran sampai pelatihan


### Loss
Pada Penelitian ini, Model mengunakan Loss sebagai berikut:
+ `MeanSquaredError(MSE)` digunakan untuk mengukur kesesuaian atau kedekatan sebuah estimator 
^θ dari parameter θ. Misalkan ^θ merupakan estimator bagi parameter θ
yang tidak diketahui dari sampel acak X1,X2,…,Xn. Dengan demikian, deviasi ^θ
terhadap nilai sebenarnya dari θ, yakni |^θ−θ|, dapat digunakan untuk mengukur kualitas estimator tersebut, atau kita bisa menggunakan (^θ−θ)2 untuk tujuan mempermudah penghitungan.
    <div><img src="https://github.com/hendradra1234/Machine-Learning/blob/master/mse.jpg?raw=true" width="300"/></div>

### Metrics
Pada Penelitian ini, Model mengunakan Metrics sebagai berikut:
+ `MAE (MeanAbsoluteError)` merupakan salah satu metrics yang digunakan untuk menghitung kualitas dari suatu model machine learning. MAE akan menghitung rata-rata dari jumlah error yang terjadi pada sebuah model.
### Callback
Tensorflow callback merupakan salah satu fungsi di dalam Tensorflow yang akan di eksekusi pada kondisi tertentu. Callback yang digunakan untuk Penelitian ini meliputi:
+ `EarlyStopping` merupakan fungsi pada tensorflow yang berfungsi untuk mencegah overfitting pada model, parameter yang digunakan untuk Penelitian ini meliputi:
    + `monitor` merupakan parameter yang digunakan untuk menentukan metric yang akan di monitor selama masa training.
    + `patience` paramater yang digunakan untuk menentukan sampai epoch ke berapa fungsi EarlyStopping menunggu sebelum melakukan eksekusi.
    + `verbose` parameter yang digunakan untuk menentukan apakah akan menampilkan pesan ke console. verbose memiliki nilai sebagai berikut:
        + `0` tidak menampilkan apapun.
        + `1` menampilkan Progress Bar.
        + `2` hanya menampilkan epoch pada saat eksekusi.

+ `CSVLogger` merupakan fungsi pada Tensorflow yang berfungsi untuk menyimpan log Training ke format CSV, parameter yang digunakan untuk penelitian ini meliputi:
    + `filename` merpakan parameter untuk menetukan nama Log file CSV hasil training.
    + `separator` merupakan paramater untuk menentukan separator yang akan digunakan pada file CSV.

### Training
Training model dilakukan dengan mengunakan batchSize dinamis, epoch dinamis dan step_per_epoch bernilai 100. BatchSize dihitung dengan mengunakan relasi jumlah memori Runtime yang tersedia, sedangkan Epoch dinamis memakai rumus:
```
epoch = ((train_len / batchsize) / steps_per_epoch) * 2
```
untuk menghitung epoch.

## Evaluation
### Evaluasi hasil Training
<div>
    <img src="https://github.com/hendradra1234/Machine-Learning/blob/master/validation_mse.png?raw=true" width="500"/>
</div>
Pengurangan Loss pada hasil Training menunjukkan bahwa model melakukan Training dengan baik dan efisien

## Model Evaluation
| parameter                              | value    |
|----------------------------------------|----------|
|Median Absolute Error (MAE)             |  197.51  |
|Mean Absolute Percentage Error (MAPE)   |  3.27 %  |
|Median Absolute Percentage Error (MDAPE)|  3.27 %  |

Dari hasil model evaluation, model memiliki parameter error sebagai berikut:
+ Median Absolute Error (MAE): MAE bernilai 197.51.
+ Mean Absolute Percentage Error (MAPE): MAPE bernilai 22,15, yang berarti rata-rata prediksi menyimpang dari nilai sebenarnya adalah sebesar 3.36%.
+ Median Absolute Percentage Error (MDAPE): MDAPE bernilai 3.36%, sehingga menunjukkan bahwa ada beberapa outlier di antara kesalahan prediksi. 50% dari prediksi menyimpang lebih dari 3.36%, dan 50% berbeda kurang dari 3,36% dari nilai sebenarnya.
### Short Term Prediction
<div>
    <img src="https://github.com/hendradra1234/Machine-Learning/blob/master/shortTermPrediction.png?raw=true" width="900"/>
</div>
Prediksi Jangka pendek dalam periode 1 tahun menunjukkan pola pergerakan saham secara cukup akurat, terdapat sedikit deviasi pada saat harga saham volatile, deviasi antara nilai sebenarnya dan nilai prediksi disebut residual.

### Medium Term Prediction
<div>
    <img src="https://github.com/hendradra1234/Machine-Learning/blob/master/mediumTermPrediction.png?raw=true" width="900"/>
</div>
Prediksi Jangka menengah dalam periode 4 tahun menunjukkan pola pergerakan saham secara cukup akurat, terdapat sedikit deviasi pada saat harga saham volatile, deviasi antara nilai sebenarnya dan nilai prediksi disebut residual.

### Long Term Prediction
<div>
    <img src="https://github.com/hendradra1234/Machine-Learning/blob/master/longTermPrediction.png?raw=true" width="900"/>
</div>

Prediksi Jangka panjang dalam periode 22 tahun menunjukkan pola pergerakan saham secara akurat, terdapat sedikit deviasi pada saat harga saham volatile, deviasi antara nilai sebenarnya dan nilai prediksi disebut residual.
### Next Day Prediction
| parameter                                                     | value         |
|---------------------------------------------------------------|---------------|
| The close price for  PT Astra Internasional at 2022-10-11 was | 6625.0        |
| The predicted close price is                                  | 6905 (+4.06%) |
Prediksi menunjukkan saham memiliki kemungkinan untuk naik pada tanggal 10 November 2022