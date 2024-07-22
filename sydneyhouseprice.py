import pandas as pd  
import numpy as np  
import seaborn as sns  
import matplotlib.pyplot as plt  
import missingno as msno  

from sklearn.impute import KNNImputer  
from sklearn.metrics import mean_squared_error 
from sklearn.model_selection import train_test_split  

# Veri setini oku
df = pd.read_csv("SydneyHousePrices.csv")   
print(df.head())  # Veri çerçevesinin ilk birkaç satırını yazdır
print(df.info())  # Veri çerçevesinin sütun bilgilerini ve veri türlerini yazdır
print(df.describe())  # Sayısal sütunların özet istatistiklerini yazdır
print(df.isnull().sum())  # Her sütundaki eksik değerlerin sayısını yazdır

# Tarih sütununu datetime formatına çevir
df["Date"] = pd.to_datetime(df["Date"])  # 'Date' sütununu datetime formatına dönüştür
df["Year"] = df["Date"].dt.year  # 'Date' sütunundan yılı çıkar ve yeni bir 'Year' sütunu oluştur
df["Month"] = df["Date"].dt.month  # 'Date' sütunundan ayı çıkar ve yeni bir 'Month' sütunu oluştur
df["Day"] = df["Date"].dt.day  # 'Date' sütunundan günü çıkar ve yeni bir 'Day' sütunu oluştur
df = df.drop(["Id", "Date"], axis=1)  # 'Id' ve 'Date' sütunlarını veri çerçevesinden kaldır

# Veri analizi
list_name = []  # Sütun isimlerini saklayacak liste
list_type = []  # Sütun veri türlerini saklayacak liste
list_total_value = []  # Sütundaki toplam değer sayısını saklayacak liste
list_missing_value = []  # Sütundaki eksik değer sayısını saklayacak liste
list_unique_value = []  # Sütundaki benzersiz değerlerin sayısını saklayacak liste

# Her sütun için bilgileri topla
for i in df.columns:
    list_name.append(i)  # Sütun adını ekle
    list_type.append(str(df[i].dtype))  # Sütun veri türünü ekle
    list_total_value.append(df[i].notnull().sum())  # Toplam mevcut değer sayısını ekle
    list_missing_value.append(df[i].isnull().sum())  # Eksik değer sayısını ekle
    list_unique_value.append(len(df[i].unique()))  # Benzersiz değerlerin sayısını ekle

# Toplanan bilgileri bir veri çerçevesine dönüştür
df_info = pd.DataFrame(data={"Total_Value": list_total_value, "Missing_Value": list_missing_value, "Unique_Value": list_unique_value, "Type": list_type}, index=list_name)
print(df_info)  # Bilgi veri çerçevesini yazdır

# Görselleştirme
df["suburb"].value_counts()[:15].plot.barh()  # 'suburb' sütunundaki en sık geçen 15 değer için yatay bar grafiği çiz
plt.show()  # Grafiği göster

df["propType"].value_counts()[:15].plot.barh()  # 'propType' sütunundaki en sık geçen 15 değer için yatay bar grafiği çiz
plt.show()  # Grafiği göster

# Sayısal sütunların dağılımını KDE grafikleriyle göster
data_num = df.select_dtypes(["float64", "int64"]).columns  # Sayısal veri türündeki sütunları seç
fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(15, 15))  # 4 satır ve 2 sütunlu bir figür oluştur
count = 0
for i in range(4):
    for j in range(2):
        if count < len(data_num):
            sns.kdeplot(df[data_num[count]], ax=ax[i][j], shade=True, color="#008080")  # KDE grafiği çiz
            count += 1
plt.show()  # Grafikleri göster

# Aylara göre araç sayısını göster
sns.countplot(x="Month", data=df)  # 'Month' sütunundaki değerlerin sayısını gösteren sayım grafiği çiz
plt.show()  # Grafiği göster

# Yıl ve fiyat ilişkisini barplot ile göster
plt.figure(figsize=(10, 5))  # Grafik boyutunu ayarla
sns.barplot(x="Year", y="sellPrice", data=df)  # 'Year' ve 'sellPrice' arasındaki ilişkiyi barplot ile göster
plt.show()  # Grafiği göster

# Ay ve fiyat ilişkisini barplot ile göster
sns.barplot(x="Month", y="sellPrice", data=df)  # 'Month' ve 'sellPrice' arasındaki ilişkiyi barplot ile göster
plt.show()  # Grafiği göster

# Yıl ve ay bazında ev fiyatlarının ortalamasını gösteren ısı haritası oluştur
heat = pd.pivot_table(data=df, index='Month', values='sellPrice', columns='Year')  # Pivot tablosu oluştur
heat.fillna(0, inplace=True)  # Eksik değerleri 0 ile doldur
plt.figure(figsize=(15, 10))  # Grafik boyutunu ayarla
plt.title('Yıllara ve Aylara Ev Fiyat Ortalamaları Isı Haritası')  # Başlık ekle
sns.heatmap(heat, annot=True, fmt=".1f", cmap="YlGnBu")  # Isı haritası grafiği çiz
plt.show()  # Grafiği göster

# Eksik değerlerin görselleştirilmesi
msno.bar(df)  # Eksik değerlerin bar grafiğini çiz
plt.show()  # Grafiği göster

msno.heatmap(df)  # Eksik değerlerin ısı haritasını çiz
plt.show()  # Grafiği göster

msno.matrix(df)  # Eksik değerlerin matrisini çiz
plt.show()  # Grafiği göster

# Bölgesel ev fiyatlarının ortalamalarını hesapla ve sırala
suburb_siniflandirma = df.groupby('suburb')['sellPrice'].mean().sort_values(ascending=False)  # 'suburb' bazında ortalama 'sellPrice' hesapla ve sırala
print(suburb_siniflandirma)  # Bölgesel ortalama ev fiyatlarını yazdır

# Sınıflama ve one-hot encoding
suburb_grup1 = list(suburb_siniflandirma[:137].index)  # İlk 137 bölgeyi birinci gruba al
suburb_grup2 = list(suburb_siniflandirma[137:274].index)  # Sonraki 137 bölgeyi ikinci gruba al
suburb_grup3 = list(suburb_siniflandirma[274:411].index)  # Sonraki 137 bölgeyi üçüncü gruba al
suburb_grup4 = list(suburb_siniflandirma[411:548].index)  # Sonraki 137 bölgeyi dördüncü gruba al
suburb_grup5 = list(suburb_siniflandirma[548:].index)  # Kalan bölgeleri beşinci gruba al

# Bölge sütununu gruplara göre değiştir
df["suburb_group"] = np.nan  # Yeni bir sütun oluştur
df.loc[df["suburb"].isin(suburb_grup1), "suburb_group"] = 0  # İlk grubu 0 ile değiştir
df.loc[df["suburb"].isin(suburb_grup2), "suburb_group"] = 1  # İkinci grubu 1 ile değiştir
df.loc[df["suburb"].isin(suburb_grup3), "suburb_group"] = 2  # Üçüncü grubu 2 ile değiştir
df.loc[df["suburb"].isin(suburb_grup4), "suburb_group"] = 3  # Dördüncü grubu 3 ile değiştir
df.loc[df["suburb"].isin(suburb_grup5), "suburb_group"] = 4  # Beşinci grubu 4 ile değiştir

df = df.drop("suburb", axis=1)  # Orijinal 'suburb' sütununu kaldır

# One-hot encoding ile kategorik değişkenleri dönüştür
df = pd.get_dummies(df, columns=["propType"], prefix=["propType"])  # 'propType' sütununu one-hot encoding ile dönüştür
print(df.head())  # Güncellenmiş veri çerçevesini yazdır

# Sayısal sütunların boxplot grafiğini çiz
data_num = df.select_dtypes(["int64", "float64"]).columns  # Sayısal veri türündeki sütunları seç
fig, ax = plt.subplots(nrows=len(data_num), figsize=(15, len(data_num) * 5))  # Her sayısal sütun için bir grafik oluştur
for i, col in enumerate(data_num):
    sns.boxplot(df[col], ax=ax[i])  # Boxplot graf



