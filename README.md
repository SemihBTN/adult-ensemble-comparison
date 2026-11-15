# Ensemble Yöntemlerinin Karşılaştırması: UCI Adult Veri Seti

Bir banka kredi kararı senaryosunda, kişinin yıllık gelirinin 50 bin doların üzerinde olup olmadığını tahmin eden modelleri karşılaştırdığım uçtan uca bir çalışma. Veri hazırlığı SQL ile, modelleme Python ile yapıldı. Bu projenin amacı sadece "en yüksek skoru bulmak" değil, **sızıntısız, tekrar üretilebilir ve gürültüden ayırt edilebilir** bir karşılaştırma kurmaktı.

## Özet

- **Veri:** UCI Adult (1994 ABD nüfus sayımı). Eğitim 32.561, test 16.281 kayıt.
- **Senaryo:** Banka kredi kararı. Yanlış "yüksek gelirli" demek pahalı.
- **Ana metrik:** `pr60` = recall %60 iken precision. Yan metrikler: recall, F1, accuracy.
- **En iyi model:** LightGBM. Test setinde recall 0,60 iken **precision 0,813**.
- **Bulgular:**
  - Boosting modelleri (Gradient Boosting, LightGBM) lojistik regresyonu ve varsayılan Random Forest'ı belirgin biçimde geçti.
  - LightGBM, Gradient Boosting'den daha iyi ve yaklaşık 8 kat daha hızlı.
  - Hiperparametre ayarı, gürültü seviyesinin üstünde bir kazanç getirmedi.
  - `sex` ve `race` sütunlarını modelden çıkarmak başarıdan hiçbir şey kaybettirmedi.

## Veri

Kaynak: <https://archive.ics.uci.edu/static/public/2/adult.zip> (`adult.data` eğitim, `adult.test` test).

Her satır bir kişi. Hedef: `income` (1 = yılda 50 bin dolardan fazla, 0 = değil). Eğitim verisinde yüksek gelirli oranı %24,1. Yani her şeye "düşük gelirli" diyen model %75,9 accuracy yapar. Bu projenin tabanı bu sayı.

Ham dosyalarda fark edilen sorunlar:

| Sorun | Çözüm |
|---|---|
| Ayraç `, ` olduğu için her metin değerin başında boşluk var | `TRIM` |
| Eksik değerler `?` olarak yazılmış (`workclass` 2.799, `occupation` 2.809, `native_country` 857, iki dosya toplamı) | `NULLIF(..., '?')` |
| `adult.test`'in ilk satırı veri değil (`\|1x3 Cross validator`) | Yüklemede atıldı (15 alanlı olmayan satırlar) |
| `adult.test` etiketlerinin sonunda nokta var (`<=50K.`) | `REPLACE(..., '.', '')` |
| Tekrar eden satırlar (eğitimde 24, testte 5) | Sadece eğitimde silindi, test seti resmi haliyle bırakıldı |

## Veri Hazırlığı (SQL / SQLite)

Üç katmanlı tablo yapısı kullandım. Ham veri hiç değiştirilmez, temizlik her seferinde ham tablodan yeniden üretilebilir.

| Tablo | İçerik |
|---|---|
| `raw_adult` | Ham veri, tüm sütunlar `TEXT`, `source` sütunu (`train`/`test`) |
| `clean_adult` | Boşluk kırpılmış, `?` → `NULL`, sayılar `INTEGER`, `income` 0/1 |
| `final_adult` | `clean_adult` + eğitimdeki tekrarlar silinmiş |

```sql
CREATE TABLE clean_adult AS
SELECT
  CAST(age AS INTEGER)               AS age,
  NULLIF(TRIM(workclass), '?')       AS workclass,
  CAST(fnlwgt AS INTEGER)            AS fnlwgt,
  TRIM(education)                    AS education,
  CAST(education_num AS INTEGER)     AS education_num,
  TRIM(marital_status)               AS marital_status,
  NULLIF(TRIM(occupation), '?')      AS occupation,
  TRIM(relationship)                 AS relationship,
  TRIM(race)                         AS race,
  TRIM(sex)                          AS sex,
  CAST(capital_gain AS INTEGER)      AS capital_gain,
  CAST(capital_loss AS INTEGER)      AS capital_loss,
  CAST(hours_per_week AS INTEGER)    AS hours_per_week,
  NULLIF(TRIM(native_country), '?')  AS native_country,
  CASE WHEN REPLACE(TRIM(income), '.', '') = '>50K' THEN 1 ELSE 0 END AS income,
  source
FROM raw_adult;

CREATE TABLE final_adult AS
SELECT DISTINCT * FROM clean_adult WHERE source = 'train'
UNION ALL
SELECT * FROM clean_adult WHERE source = 'test';
```

SQL'de yalnızca **satır bazlı, veriden bir şey öğrenmeyen** işlemler yapıldı (boşluk silme, `?` → `NULL`, tür çevirme). Eksik değer doldurma, ölçekleme ve kodlama gibi veriden öğrenen işlemler sızıntı olmaması için Python tarafında, CV içinde yapıldı.

Keşif sorgularından örnek: yüksek gelirli oranı eğitim düzeyine göre okul öncesinde %0'dan doktorada %74'e kadar düzenli artıyor. `education` ile `education_num` birebir aynı bilgiyi taşıyor.

## Özellik Kararları

Modelde 10 özellik var: `age`, `workclass`, `education_num`, `marital_status`, `occupation`, `relationship`, `capital_gain`, `capital_loss`, `hours_per_week`, `native_country`.

| Atılan sütun | Sebep |
|---|---|
| `fnlwgt` | Anketin örnekleme ağırlığı, kişinin özelliği değil. Gerçek bir başvuruda bulunmaz. |
| `education` | `education_num` ile birebir aynı bilgi. |
| `sex`, `race` | Kredi senaryosunda bu sütunlarla karar vermek etik ve yasal sorun. Başarıya etkisi ayrıca ölçüldü (aşağıda). |

## Yöntem

- **Sızıntı önleme:** Eksik doldurma, ölçekleme ve kategori kodlama `Pipeline` + `ColumnTransformer` içinde. CV'nin her katında sadece o katın eğitim kısmından öğreniliyor. Test setine model seçimi boyunca hiç dokunulmadı.
  - Sayısal sütunlar: medyan ile doldurma + `StandardScaler`
  - Metin sütunları: boşlara `"Unknown"` + `OneHotEncoder(handle_unknown="ignore")`
- **Çapraz doğrulama:** `RepeatedStratifiedKFold` (5 kat × 3 tekrar = 15 skor, `random_state=42`). Tüm modeller aynı 15 katta değerlendirildi, bu sayede kat bazında eşleştirilmiş karşılaştırma yapılabildi.
- **Ana metrik `pr60`:** Precision ve recall birbirinin aleyhine çalışır (eşiği yükseltince precision artar, recall düşer). Modelleri adil karşılaştırmak için her katın precision-recall eğrisinde recall'un en az 0,60 olduğu noktadaki en yüksek precision alındı.
- **"İyileşti" kriteri:** Fark, CV skorlarının standart sapmasından büyük olmadıkça iyileşme sayılmadı. Bu kural aşağıdaki tüm yorumlarda uygulandı.

## Sonuçlar (5×3 CV, eğitim verisi)

| Model | accuracy | precision | recall | F1 | **pr60** | pr60 std | Süre (sn) |
|---|---|---|---|---|---|---|---|
| Çoğunluk sınıfı | 0,7591 | 0 | 0 | 0 | tanımsız | | |
| Lojistik Regresyon | 0,8512 | 0,7339 | 0,6002 | 0,6603 | 0,7356 | 0,0144 | 0,20 |
| Random Forest (100 ağaç) | 0,8480 | 0,7106 | 0,6229 | 0,6638 | 0,7253 | 0,0145 | 2,75 |
| Gradient Boosting | 0,8661 | 0,7887 | 0,6067 | 0,6858 | 0,7930 | 0,0139 | 2,23 |
| **LightGBM** | **0,8731** | 0,7818 | 0,6570 | 0,7139 | **0,8152** | 0,0109 | 0,29 |
| LightGBM (ayarlı) | 0,8736 | 0,7817 | 0,6598 | 0,7156 | 0,8206 | 0,0114 | 0,63 |

Notlar:
- `precision` ve `recall` sütunları varsayılan 0,5 eşikte hesaplandı. Modellerin recall'u birbirinden farklı olduğu için bu iki sütun birbirine kıyaslanmamalı, karşılaştırma `pr60` ile yapıldı.
- Süreler bir katın ortalama eğitim süresi. Random Forest tüm çekirdeklerle (`n_jobs=-1`) çalıştı, diğerleri tek çekirdekle, yani Random Forest'ın gerçek maliyeti tabloda göründüğünden yüksek.
- Çoğunluk sınıfı modeli hiç pozitif tahmin etmediği için precision tanımsız (0/0).

**Yorum:**
- Lojistik regresyon ile varsayılan Random Forest arasındaki fark (0,010) std'nin (0,014) altında: ikisi eşit.
- Gradient Boosting, lojistiği yaklaşık 0,057 farkla geçti (std'nin ~4 katı), yani gerçek bir iyileşme.
- LightGBM, Gradient Boosting'e göre `pr60`'ta ortalama +0,022 daha iyi (kat farkının std'si 0,009) ve 15 katın 15'inde önde.

## Hiperparametre Ayarı

`RandomizedSearchCV`, 30 deneme × 15 kat, puanlama `pr60`. Aranan aralıklar 32 bin satırlık veriye göre dar tutuldu (`n_estimators` 100-600, `learning_rate` 0,02-0,2, `num_leaves` 8-64, `min_child_samples` 10-100, `subsample` 0,6-1,0, `colsample_bytree` 0,5-1,0, `reg_lambda` 0,01-10, `subsample_freq=1`).

En iyi bulunan ayar: `n_estimators=448`, `learning_rate=0,064`, `num_leaves=10`, `min_child_samples=89`, `subsample=0,87`, `colsample_bytree=0,58`, `reg_lambda=0,59`.

Ayarlı ve ayarsız LightGBM'in kat bazında eşleştirilmiş farkı: ortalama +0,0054, farkın std'si 0,0053, 11/15 katta ayarlı önde. Üstelik ayarlar aynı katlarla seçildiği için bu fark iyimser. **İyileşme kanıtlanamadı**, bu yüzden final model olarak daha sade ve iki kat hızlı olan **ayarsız LightGBM** seçildi.

## Test Seti Sonucu

Test setine yalnızca bir kez, en sonda bakıldı.

1. Eşik sadece eğitim verisinden seçildi: 5 katlı stratified CV ile her satır için modelin görmediği olasılık üretildi, recall'u en az 0,60 yapan en yüksek eşik alındı → **0,565** (eğitimde precision 0,8126, recall 0,6001).
2. Model tüm eğitim verisiyle eğitildi, test setine bu eşik uygulandı.

| Metrik | Test seti |
|---|---|
| precision | **0,8129** |
| recall | 0,5998 |
| accuracy | 0,8729 |
| F1 | 0,6903 |

CV'de ölçülen `pr60` (0,815) ile test precision'ı (0,813) neredeyse aynı: model ezber yapmamış, eşik seçimi test setine taşınmış. Test setinde 3.846 yüksek gelirli kişi olduğu için sonuçlarda yaklaşık ±0,015 doğal oynama payı var.

## Deney: `sex` ve `race` Modele Girerse?

Tek değişiklik: bu iki sütun eklendi (aynı katlar, ayarsız LightGBM).

| | pr60 |
|---|---|
| `sex` + `race` ile | 0,8151 ± 0,0095 |
| `sex` + `race` olmadan | 0,8152 ± 0,0109 |

Ortalama fark −0,0001, 15 katın 8'inde ekleme daha iyi. Bu sütunlar modele ek bir bilgi katmıyor. Bunun sebebi aynı bilginin `relationship` (Husband/Wife), `marital_status`, `occupation` ve `hours_per_week` içinde zaten bulunması.

## Sınırlılıklar

- Veri 1994'e ait. Gelir eşiği ve dağılımlar bugünkü koşulları yansıtmaz. Kredi senaryosu projeye çerçeve vermek için kuruldu, gerçek banka verisi değil.
- `sex` ve `race` çıkarıldı ama yerine geçen vekil sütunlar (özellikle `relationship`) duruyor. Bu, modelin cinsiyetten tamamen bağımsız olduğu anlamına gelmez. Gerçek bir adalet analizi ayrı bir çalışma gerektirir.
- Hiperparametre araması dar ve 30 denemeyle sınırlı. Random Forest hiç ayarlanmadı.
- Test sonucu tek bir ölçüm, güven aralığı yaklaşık ±0,015.
- Çoğunluk modeli dışında bir "iş değeri" (yanlış alarmın parasal maliyeti) hesaplanmadı. Recall %60 hedefi bir tasarım tercihi, matematiksel bir zorunluluk değil.

## Tekrar Üretme

```bash
pip install pandas scikit-learn lightgbm scipy
```

1. `adult.zip`'i indirip `adult.data` ve `adult.test` dosyalarını proje klasörüne koyun.
2. Veri hazırlığı notebook'unu çalıştırın: `raw_adult` → `clean_adult` → `final_adult` (`adult.db` dosyası oluşur).
3. Modelleme notebook'unu çalıştırın: baseline, dört model, ayar, test değerlendirmesi, `sex`/`race` deneyi.

Veri dosyaları depoya eklenmedi, UCI linkinden indirilir.

## Sonraki Adımlar

- CatBoost ve XGBoost'u aynı katlarda tabloya eklemek.
- `relationship` ve `marital_status` çıkarılarak vekil sütunların etkisini ölçmek.
- Hata maliyetine dayalı bir eşik seçimi (yanlış alarm ve kaçırılan müşteri için ayrı maliyetler).
