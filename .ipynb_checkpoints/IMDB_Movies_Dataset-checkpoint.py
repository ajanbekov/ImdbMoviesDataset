import pandas as pd

df = pd.read_csv("imdb_top_1000.csv")
print(df.head()) # the first 5 lines
print(df.shape) # number of rows and columns
print(df.columns) # sütun isimleri (column names)
print(
    "\nBefore cleaning, data info:\n", df.dtypes
)
print(
    "\n The number of missing values \n", df.isna().sum()
)
print(df.info()) # Data types and missing value status
# non-null = no missing (empty) value at all


# DATA CLEANING AND PREPARATION STAGE (PANDAS)

df.columns = df.columns.str.strip() # Clear unnecessary gaps and format errors

# Fill missing values in the "Certificate" column with "unknown"
if "Certificate" in df.columns: # Check if the column exists
    df["Certificate"] = df["Certificate"].fillna( # Fill missing values
        "unknown"
    )

# # Fill missing values with the mean of the column
if "Meta_score" in df.columns:
    df["Meta_score"] = df["Meta_score"].fillna(
        df["Meta_score"].mean() # Fill missing values with the mean of the column (ortalama ile doldurma)
    )

# Convert the "Gross" column to numbers (remove commas, replace missing values with 0)
if "Gross" in df.columns:
    df["Gross"] = df["Gross"].fillna("0") # Fill missing values with 0 / df["Gross"] = df["Gross"] iki kere yazmazsak değişiklik olmaz
    df["Gross"] = df["Gross"].astype(str).str.replace(",", "").astype(float) # Remove commas and convert to float

# Convert the "Released_Year" column to numeric, setting errors to NaN
if "Released_Year" in df.columns:
    df["Released_Year"] = pd.to_numeric( # Convert to numeric
        df["Released_Year"],
        errors="coerce" # If conversion fails, set as NaN
    )

# Filling NaN's with the average
mean_year = df["Released_Year"].mean()
df["Released_Year"] = df["Released_Year"].fillna(
    mean_year
)

df.Released_Year = df.Released_Year.astype('int64')

# Convert Certificate's data type to category
colomuns = ["Certificate", "Genre"]
for i in colomuns:
    df[i] = df[i].astype("category")

# Changing datatype of runtime to int
df.Runtime = df.Runtime.str.replace(" min", "").astype('int64')

print(
    "\nAfter cleaning, data info:\n", df.dtypes
)
print(
    "\n The number of missing values \n", df.isna().sum()
)
print(
    "\nAfter cleaning, data info:\n", df.info()
)


# DATA VISUALIZATION STAGE (MATPLOTLIB)

# Distribution of films by year with bar chart
import matplotlib.pyplot as plt
df2 = df.groupby("Released_Year").size() # Count number of films per year
df2.plot(
    kind="bar", # Çubuk grafik (bar chart)
    figsize=(12, 6)
)
plt.xticks( # Set x-ticks to show every 5th year
    range(
        0, len(df2), 5
    ),
    df2.index[::5] # Show every 5th year
)
plt.xlabel(
    None
)
plt.ylabel(
    "Number of Films"
)
plt.title(
    "Distribution of Films by Year"
)
plt.show()
print(
    "\nYears With Most Films:\n",
    df2.nlargest(2). # nlargest() = returns the largest (n) values / en büyük 2 yılı verir
    rename_axis(None). # İndeks adını kaldırma
    to_string()
)

# Most Common Genres
s = (
    df["Genre"].dropna(). # dropna() = remove missing values
    str.split(", "). # türleri virgül ile ayırır
    explode() #  her türü ayrı bir satırda gösterir
)
genres = s.value_counts().to_dict() # türlerin sayısını sayar ve sözlüğe çevirir
plt.figure(
    figsize=(10,7)
)
plt.bar(
    genres.keys(), # tür isimleri
    genres.values() # tür sayıları
)
plt.xticks(
    rotation=90, fontsize=9 # x eksenindeki yazıları 90 derece döndürme
)
plt.title(
    "Most Common Genres"
)
plt.show()

# Actors Appear In Top Movies
columns = [
    "Star1", "Star2", "Star3", "Star4"
]
s = (df[columns].stack(). # stack() = sütunları satırlara dönüştürür
     str.strip()) # strip() = önde ve sonda boşlukları temizler
stars = s.value_counts().head(10).to_dict() # en çok oynayan 10 aktörün sayısını sayar ve sözlüğe çevirir
plt.figure(
    figsize=(10,7)
)
plt.bar(
    stars.keys(),
    stars.values()
)
plt.xticks(
    rotation=35, fontsize=8
)
plt.title(
    "Actors Appear In Top Movies"
)
plt.show()

# Directors With The Highest Average on IMDB
df2 = df.Director.value_counts() # Yönetmenlerin film sayısını sayar (Counts number of films directors)
df2 = df2[ # sadece birden fazla filmi olan yönetmenleri alır (only directors with multiple films)
    df.Director.value_counts() > 1
].index

df2 = df[ #
    df.Director.isin(df2) # isin() = filtreleme işlemi yapar (making a filter)
].groupby( # yönetmenlere göre gruplar
    "Director",
    observed=True # kategorik veriyle çalışırken gereksiz grupların oluşmasını engeller, daha hızlı, temiz sonuçlar elde edilir
)[
    "IMDB_Rating"
].mean().round(2).nlargest(10) # round(2) = virgülden sonra 2 basamak gösterir

ax = df2.plot( # ax'a atayarak grafikle oynanabilir (ax.text() ile çubukların üzerine değer yazmak gibi)
    kind = "bar",
    figsize=(8,7)
)
plt.xticks(rotation=35, fontsize=8)
for i, value in enumerate(df2): # çubukların üzerine değer yazmak için / enumerate() = index ve value döner
    ax.text(
        i, value - 0.5, # value - 0.5 ile değeri çubuğun biraz altına yazdırır
        str(value), # değeri stringe çevirir
        ha = "center", # horizontal alignment
        fontsize = 7,
        va = "top", # vertical alignment
        color = "white"
    )
plt.show()


# GENRE CLASSIFICATION WITH SCIKIT-LEARN

y = df["Genre"] # hedef sutun (target column)
x = df[ # tahmin için faydalı sutunlar (features)
    [
        "IMDB_Rating", "Meta_score", "Gross", "Runtime", "Released_Year"
    ]
]
y = y.str.split(",").str[0].str.strip() # sadece ilk türü alır ve boşlukları temizler

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split( # veriyi eğitim ve test olarak ayırır
    x, y,
    test_size=0.2, # %20 test verisi
    random_state=42, # veri bölmenin her seferinde aynı şekilde gerçekleşmesini ve sonuçların tekrarlanabilir olmasını sağlar
)

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Eksik verileri doldurma (ortalama ile)
imputer = SimpleImputer(
    strategy="mean" # eksik değerleri ortalama ile doldurur
)
x_train = imputer.fit_transform( # fit_transform() = eğitim verisinin ortalamasını öğrenip doldurur
    x_train
)
x_test = imputer.transform( # imputer = eksik (NaN) değerleri belirli bir stratejiye göre doldurur
    x_test
)

# Ölçekleme (Standardization) = tüm özellikleri eşit şekilde değerlendirmesi için
scaler = StandardScaler()
x_train = scaler.fit_transform(
    x_train
)
x_test = scaler.transform(
    x_test
)

# Model Kurulumu ve Eğitimi (Model Installation and Training)
from sklearn.ensemble import RandomForestClassifier # ensemble =
model = RandomForestClassifier(
    n_estimators=100, # modelin kullanacağı bağımsız karar ağacı sayısı
    # 100 verildiğinde eğitim süresi makul kalır, modelin doğruluğu yeterli olur
    random_state=42
)
model.fit(
    x_train,
    y_train
)

# Tahmin ve Doğruluk Oranına Bakma (Check Prediction and Accuracy Rate)
from sklearn.metrics import accuracy_score, classification_report # accuracy_score = Doğruluk oranı hesaplamak için
# classification_report = Sınıflandırmanın ayrıntılı raporunu almak için

y_pred = model.predict(
    x_test
)
print(
    "\n Doğruluk Oranı (Accuracy): ", round(
        accuracy_score(y_test, y_pred), 3 # gerçek etiketler (y_test) modelin tahminleri (y_pred) ne kadar örtüştüğünü oransal olarak bulur

    )
)
print(
    "\nSınıflandırma Raporu (Classification Report): \n", classification_report(
        y_test, y_pred,
        zero_division=0
    )
)

# Karışıklık Matrisi (Confusion Matrix)
from sklearn.metrics import ConfusionMatrixDisplay # karışıklık matrisi (confusion matrix) için hazır bir görselleştirme aracı
ConfusionMatrixDisplay.from_estimator(
    model,
    x_test,
    y_test,
    normalize='true', # 1.0, 0.9, 0.8 → büyük kısmı doğru tahmin edilmiş / 0.0, 0.1 → Model doğru tahmin edememiş
    xticks_rotation=90
)
plt.title(
    "Karışıklık Matrisi (Confusion Matrix) - Genre Classification"
)
plt.show() # the film belongs to more than one genre that's why accurancy rate is low