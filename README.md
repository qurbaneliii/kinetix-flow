## Bakı Metropolitenində Sərnişin Axınının Kinetix AI ilə Proqnozlaşdırılması

### Məlumatların (Data) Toplanması, Təhlili və Tətbiqi

### Data toplanması

Hər hansı bir Maşın Öyrənməsi (Machine Learning) modelini ərsəyə gətirmək üçün atılacaq ilk və ən həlledici addım **doğru məlumatların (data) tapılması və toplanmasıdır**. Çünki necə ki, bir avtomobilin hərəkət etməsi üçün yanacaq lazımdır, ML modellərinin də "öyrənməsi" və doğru qərarlar verə bilməsi üçün keyfiyyətli tarixçəyə ehtiyacı var.

Bəs bu proses praktikada necə işləyir?

Maşın Öyrənməsi modelləri, əslində, illərin təcrübəsini qısa müddətdə analiz edə bilən yorulmaz bir mütəxəssis kimi davranır. Sistem işə düşdükdə aşağıdakı məntiqi ardıcıllığı izləyir:

- **Keçmişdən öyrənmə:** Model ilk olaraq əvvəlki aylara və illərə aid böyük həcmli sərnişin məlumatlarını oxuyur. O, bu rəqəmlər arasındakı gizli qanunauyğunluqları, təkrarlanan trendləri və statistik asılılıqları kəşf edir.
- **Xüsusiyyətlərin (Features) təhlili:** Sıxlıq sadəcə quru rəqəmlərdən ibarət deyil. Model öyrənir ki, həftənin günləri, saatlar, hava şəraiti, bayram və ya iş günləri sərnişin axınına necə təsir göstərir.
- **Proqnozlaşdırma:** Keçmişin "təcrübəsini" tam mənimsədikdən sonra model artıq bu günə fokuslanır. Mövcud vəziyyəti (hazırkı saatı, günü və s.) dəyərləndirərək, statistik hesablamalar əsasında real zamanlı bir nəticə çıxarır.

Məhz bu zəncirvari prosesin sayəsində sistem yüksək dəqiqliklə təxmin edə bilir ki: *"Bu gün, bu saatda, filan stansiyada sərnişin sıxlığı və yüklənmə səviyyəsi necə olacaq?"*

1. Kinetix AI layihəsi üzərində işə başlayarkən ilk və ən vacib addımımız doğru məlumatları (datanı) tapmaq idi. Bakı öz nəqliyyat infrastrukturu, sərnişin yüklənməsi və insanların nəqliyyat vərdişləri baxımından olduqca unikal bir şəhərdir. Buna görə də qlobalda mövcud olan hər hansı standart bir məlumat dəstini (dataset) götürüb layihəmizə tətbiq etmək effektiv olmazdı. Azərbaycan reallığında sərnişinlərin davranış modelini yalnız yerli və real məlumatlar dəqiq əks etdirə bilərdi.

Sırf Bakı Metropoliteninə uyğun məlumat axtarışımız uğurla nəticələndi. Belə ki, İnnovasiya və Rəqəmsal İnkişaf Agentliyinin idarə etdiyi **opendata.az** (Açıq Məlumatlar Portalı) üzərindən "Bakı Metropoliteni" QSC tərəfindən yerləşdirilmiş rəsmi məlumat dəstini əldə etdik. Bu məlumat dəsti 2025-ci ilin əvvəlindən 2026-cı ilin 1 mart tarixinədək olan dövrdə stansiyalar üzrə gündəlik sərnişin axınını tam əhatə edir.

**Məlumatların Təbiəti və Toplanma Metodologiyası**

Bəs bu rəqəmlər necə formalaşır və nələri ifadə edir?

Gediş sayları sərnişinlərin metro stansiyalarının giriş turniketlərində etdikləri ödənişlər — həm fiziki kartlar, həm də mobil cihazlar vasitəsilə edilən NFC əsaslı təmassız ödənişlər əsasında qeydə alınır. Lakin burada nəzərə alınmalı vacib bir nüans var: hazırda Bakı Metrosunun çıxış turniketlərində təkrar identifikasiya (kart oxudulması) aparılmır.

Bu səbəbdən, əlimizdəki məlumatlar sərnişinlərin tam səfər zəncirini (A nöqtəsindən B nöqtəsinə getdiyini) deyil, sırf stansiyalar üzrə **giriş intensivliyini və formalaşan ilkin yüklənməni** göstərir.

**Məlumat Dəstinin (Dataset) Texniki Profili:**

İstifadə etdiyimiz məlumat dəstinin əsas göstəriciləri bunlardır:

- **Əhatə dairəsi:** Bakı şəhəri
- **İştirakçı / Mənbə:** "K Group" MMC (BakıKart)
- **Yenilənmə tezliyi:** Ayda 1 dəfə
- **Məlumatın strukturu (Sütunlar):**
    - **Tarix:** Gün.ay.il formatında qeyd olunmuş zaman.
    - **Stansiya:** Bakı Metropoliteninə aid stansiyanın adı.
    - **Gündəlik gediş sayı:** Verilmiş gün ərzində həmin stansiyadan daxil olan sərnişinlərin ümumi sayı.

### **Məlumatların İşlənməsi və Hazırlıq Mərhələsi (Data Preprocessing)**

Uyğun məlumat mənbəyini müəyyən etdikdən sonra qarşımızda duran növbəti əsas məqsəd bu datanın işçi mühitə yüklənməsi, xətaların təmizlənməsi, manipulyasiyası və nəhayət, maşın öyrənməsi modeli üçün tam hazır hala gətirilməsi idi.

Açıq Məlumatlar Portalı məlumatları əldə etmək üçün bizə iki fərqli imkan təklif edirdi: API (Application Programming Interface) vasitəsilə sistemə birbaşa qoşulmaq və ya məlumatları ənənəvi CSV faylı şəklində yükləmək.

**API-dən CSV-yə Keçid: Texniki Qərarımız**
İlkin yanaşma olaraq, məlumat axınını maksimal dərəcədə avtomatlaşdırmaq məqsədilə datanı API vasitəsilə birbaşa portaldan çəkdik və üzərində ilkin əməliyyatlara başladıq. Lakin layihənin daha mürəkkəb mərhələsi olan "Feature Engineering" (yeni xüsusiyyətlərin yaradılması) prosesinə keçdikdə, API-nin dinamik təbiəti bəzi ləngimələr və struktur uyğunsuzluqları yaratmağa başladı.

Modelimizin daha stabil, sürətli və tam nəzarət edilə bilən bir infrastrukturda öyrədilməsini təmin etmək üçün praktik bir addım atdıq. API bağlantısını kənara qoyaraq, datanı statik CSV formatında endirdik. Ardından, prosesi Jupyter Notebook mühitinə daşıyaraq Python-un məlumat analizi alətləri vasitəsilə faylı oxutduq və datanı asanlıqla manipulyasiya edə biləcəyimiz strukturlaşdırılmış bir *DataFrame* halına gətirdik. Artıq məlumat dəstimiz ən dərin analizlər və modelin qurulması üçün tam etibarlı vəziyyətdə idi.

### **Məlumatların Təmizlənməsi və Standartlaşdırılması (Data Cleaning)**

Məlumatları (dataset) işçi mühitimizə yüklədikdən sonra dərhal model qurmağa keçə bilməzdik. Çünki xam (raw) məlumatlar adətən daxilində xətalar, boşluqlar və fərqli formatlar barındırır. Buna görə də, növbəti mərhələmiz Kinetix AI-ın başa düşəcəyi qüsursuz və standartlaşdırılmış bir struktur yaratmaq oldu.

Yazdığımız xüsusi alqoritm (kod) vasitəsilə bu təmizləmə prosesini bir neçə addımda avtomatlaşdırdıq:

**1. Sütun Adlarının Standartlaşdırılması**

- **Nə etdik?** Məlumat dəstindəki sütun adlarındakı böyük-kiçik hərfləri, lazımsız boşluqları və xüsusi simvolları təmizləyərək vahid bir formata (alt xəttli və kiçik hərflərlə) gətirdik. Həmçinin, sistemin sütun adlarını (İngilis və ya Azərbaycan dilində olmasından asılı olmayaraq) avtomatik tanıması üçün xüsusi axtarış məntiqi qurduq.
- **Niyə etdik?** Gələcəkdə portaldan yeni datalar çəkilərkən sütun adlarında ola biləcək hər hansı format dəyişikliyinin (məsələn, "Gediş Sayı" əvəzinə "gedis sayi" yazılması) sistemimizi çökdürməsinin qarşısını almaq üçün.

**2. Data Tiplərinin Doğrulanması (Formatting)**

- **Nə etdik?** Tarix sütununu sistemin anlaya biləcəyi rəsmi `datetime` (zaman) formatına çevirdik. Stansiya adlarının ətrafındakı görünməyən boşluqları kəsib atdıq. Sərnişin sayını isə sırf riyazi əməliyyatlar aparıla bilməsi üçün ədədi (numeric) formata keçirdik.
- **Niyə etdik?** Maşın öyrənməsi modelləri riyazi məntiqə əsaslanır. Tarixin mətn kimi yox, zaman kimi, sərnişin sayının isə sırf rəqəm kimi oxunması zaman oxu (time-series) üzrə analizlər aparmaq üçün ən vacib şərtdir.

**3. Boşluqların Ağıllı Bərpası (Linear Interpolation)**

- **Nə etdik?** Məlumatları əvvəlcə stansiyalar və tarixlər üzrə xronoloji ardıcıllıqla sıraladıq. Daha sonra əgər hər hansı bir stansiyada hansısa gün üzrə məlumat qeydə alınmayıbsa (boşluq varsa), bu boşluğu "Xətti İnterpolyasiya" (Linear Interpolation) metodu ilə doldurduq.
- **Niyə etdik?** Boş (NaN) dəyərlər süni intellekt modellərinin ən böyük düşmənidir və hesablamanı pozur. İnterpolyasiya vasitəsilə boş qalan günü sıfırla doldurmaq və ya silmək əvəzinə, sistemə həmin gündən əvvəlki və sonrakı günlərin sərnişin sayına baxaraq ən məntiqi rəqəmi riyazi olaraq təxmin edib yerinə yazmağı öyrətdik. Bu, datanın ümumi trendinin pozulmasının qarşısını aldı.

Nəticə etibarilə, bu təmizləmə əməliyyatından sonra əlimizdə Kinetix AI-ın öyrənmə prosesi (model training) və yeni xüsusiyyətlərin çıxarılması (feature engineering) üçün tam hazır, xətasız və riyazi cəhətdən dayanıqlı bir Dataframe formalaşdı.

### **Feature Engineering**

Təmizlənmiş data modelin işləməsi üçün yetərli olsa da, yüksək dəqiqlik (accuracy) əldə etmək üçün kifayət deyildi. Süni intellektə sərnişinlərin davranış modelini, şəhərin ritmini və təkrarlanan vərdişləri başa salmaq lazım idi. Bu məqsədlə xüsusi bir `KinetixFeatureEngineer` alqoritmi (sinfi) yazaraq datamızı aşağıdakı məntiqi amillərlə zənginləşdirdik:

**1. Zaman və Təqvim Xüsusiyyətlərinin Çıxarılması (Temporal Features)**

- **Nə etdik?** Sadə "Tarix" sütununu parçalayaraq yeni sütunlar yaratdıq: həftənin hansı günüdür, ayın neçəsidir, iş günüdür yoxsa həftəsonu.
- **Niyə etdik?** Çünki Bakı metrosunda bazar ertəsi səhər saatlarındakı axınla bazar gününün axını tamamilə fərqlidir. Model bu fərqi artıq riyazi olaraq görə bilir.

**2. Yerli Kontekst və Təhsil Mövsümü (Local Context)**

- **Nə etdik?** Kodun içinə xüsusi olaraq Azərbaycandakı rəsmi qeyri-iş günlərini (Novruz tətili, Yeni il və s.) və ölkəmizin akademik təqvimini (payız və yaz semestrləri) daxil etdik.
- **Niyə etdik?** Sərnişin axınının böyük bir qismi tələbələrdən ibarətdir. Modelin tətildə olan günlərlə dərs günlərini ayırd etməsi qəfil sıxlıq düşüşlərini və ya artışlarını doğru təxmin etməsi üçün kritik əhəmiyyət daşıyır.

**3. Stansiyaların Xarakteristikası (Station Typology)**

- **Nə etdik?** Bütün stansiyaları eyni tərəziyə qoymadıq. Alqoritmə öyrətdik ki, "28 May" və ya "Memar Əcəmi" tranzit/keçid (transfer hub), "Elmlər Akademiyası" və "Gənclik" isə tələbə mərkəzli (student hub) stansiyalardır.
- **Niyə etdik?** Bu təsnifat modelə hər stansiyanın spesifik yüklənmə səbəbini anlamağa və fərdi ssenarilər qurmağa kömək edir.

**4. Keçmişin Yaddaşı və Gecikmə Xüsusiyyətləri (Lag & Rolling Features)**

- **Nə etdik?** Ən vacib ML gedişlərimizdən birini edərək "Zaman Sıraları" (Time-Series) məntiqini tətbiq etdik. Modelə hər gün üçün 3 yeni sualın cavabını əlavə etdik:
    - *Dünən eyni vaxtda sərnişin sayı nə qədər idi? (lag_1_day)*
    - *Düz 1 həftə əvvəl eyni gündə vəziyyət necə idi? (lag_7_days)*
    - *Son 3 günün orta axın trendi nədir? (rolling_mean_3_days)*
- **Niyə etdik?** Süni intellekt bu "keçmişə baxış" rəqəmlərini oxuyaraq bugünkü sərnişin sayını daha inamla proqnozlaşdırır. Məsələn, əgər son 3 gündə artım trendi varsa, model bugünü də o trendə uyğun hesablayır.

**5. Maşın Dili üçün Rəqəmsallaşdırma (Encoding)**

- **Nə etdik?** Modellər hərfləri və sözləri deyil, rəqəmləri anlayır. Ona görə də stansiya adlarını xüsusi kodlaşdırma (One-hot encoding) vasitəsilə sıfır və birlərdən (0, 1) ibarət riyazi matrislərə çevirdik.

**Yekun Nəticə:** Bu prosesin sonunda bizim əlimizdə sadəcə "Tarix" və "Gediş Sayı" olan kasıb bir cədvəl yox, Kinetix AI-ın hər bir sərnişinin addımını və şəhərin nəbzi ilə ayaqlaşan qaydalarını anlaya biləcəyi 14 fərqli xüsusiyyətdən ibarət çox zəngin bir **"Öyrənmə Matrisi" (Feature Matrix)** formalaşdı.

### **Chronological Train-Test Split**

Süni intellekt modelinin (Kinetix AI) öyrənmə prosesinə başlamazdan əvvəl, əlimizdəki zənginləşdirilmiş məlumat dəstini iki hissəyə ayırmalı idik: Modelin keçmiş təcrübələrdən "öyrənməsi" üçün **Təlim (Train)** və öyrəndiklərini gələcək üzərində sınaması üçün **Test** məlumatları.

Bu mərhələdə standart təsadüfi bölünmə (random split) yerinə, sırf zaman ardıcıllığına əsaslanan çox qəti bir xronoloji bölünmə (Strict Chronological Split) arxitekturası qurduq:

**1. Qəti Zaman Sıralaması (Strict Temporal Sorting)**

- **Nə etdik?** Bütün məlumatları ilk gündən son günə (və saata) qədər xronoloji ardıcıllıqla düzdük.
- **Niyə etdik?** Zaman sıralı (Time-series) məlumatlarda keçmiş gələcəyə təsir edir. Məlumatların qarışmaması və ardıcıllığın qorunması sistemin düzgün trend tutması üçün şərtdir.

**2. Hədəfin və Öyrənmə Matrisinin Ayrılması (Target & Feature Matrix)**

- **Nə etdik?** Təxmin etmək istədiyimiz əsas hədəfi — sərnişin sayını (`y`) sistemdən ayırdıq. Yerdə qalan bütün məlumatlardan isə modelin riyazi olaraq anlaya bilməyəcəyi mətn tipli sütunları (orijinal tarix, stansiya adları) sildik və yalnız əvvəlki mərhələdə hazırladığımız rəqəmsal xüsusiyyətləri (`X`) saxladıq.
- **Niyə etdik?** Modelə "cavabı" (`y`) və "ipuçlarını" (`X`) ayrı-ayrı verməliyik ki, o, ipuçlarına baxaraq cavabı tapmağı öyrənsin. Həmçinin, alqoritmlər yalnız rəqəmlərlə işlədiyi üçün təmizlənmiş bir "riyazi matris" formalaşdırmaq məcburidir.

**3. Xronoloji 80/20 Bölünməsi (Index-Based Splitting)**

- **Nə etdik?** Məlumatları təsadüfi olaraq deyil, indeks (zaman) oxu üzrə kəsdik. İlk 80%-lik xronoloji hissəni modelin öyrənməsi (`Train`), son 20%-lik hissəni (gələcəyi) isə modelin imtahan edilməsi (`Test`) üçün ayırdıq.
- **Niyə etdik? (Data Leakage-in qarşısının alınması):** Əgər datanı təsadüfi bölsəydik, model gələcəkdəki bir hadisənin məlumatını alıb keçmişi proqnozlaşdırmaq üçün istifadə edə bilərdi. Buna Data Science-da "Məlumat sızması" (Data Leakage) deyilir. Xronoloji bölünmə ilə biz modelin qətiyyən "gələcəyə baxmasının" qarşısını aldıq. Kinetix AI yalnız keçmişi görərək gələcəyi təxmin etməyə məcbur edildi.

**4. Validasiya və Yoxlanış (Validation Check)**

- **Nə etdik?** Kodun sonuna xüsusi bir yoxlanış bloku əlavə etdik. Bu blok Təlim məlumatlarının bitdiyi tarixlə Test məlumatlarının başladığı tarixi müqayisə edərək aralarında heç bir kəsişmənin olmadığını (Strict non-overlap check) təsdiqlədi.

**Yekun Nəticə:** Artıq əlimizdə tam təhlükəsiz, zaman məntiqinə zidd olmayan, gələcəyə sızma ehtimalı sıfıra endirilmiş mükəmməl bir öyrənmə mühiti (Train) və görünməz sınaq mühiti (Test) var. Model artıq öyrədilməyə (Training) tam hazırdır.

### **Model Selection – Why XGBoost?**

**Preprocessing** və **Feature Engineering** mərhələlərindən sonra qarşımızda duran ən kritik sual bu idi: Hansı alqoritm Bakı metrosunun bu mürəkkəb sərnişin axınını ən dəqiq şəkildə proqnozlaşdıra bilər? Müxtəlif klassik və müasir yanaşmaları analiz etdikdən sonra **XGBoost (Extreme Gradient Boosting)** modelini seçdik. Bu seçimin arxasında duran əsas səbəblər bunlardır:

**1. Handling Non-linear Relationships**
Bakı metrosundakı sərnişin sıxlığı sadə, xətti bir artımla baş vermir. Məsələn, "Yağışlı hava + Monday + Academic Season" kombinasiyası sərnişin sayına qeyri-adi dərəcədə təsir edə bilər. XGBoost bu cür mürəkkəb və bir-biri ilə kəsişən faktorlar arasındakı **non-linear** əlaqələri tutmaqda dünyada ən uğurlu alqoritmlərdən biri hesab olunur.

**2. Robust Performance for Time-Series Data**
XGBoost, bizim hazırladığımız **lag features** və **rolling means** xüsusiyyətlərini çox effektiv emal edir. O, keçmişdəki trendləri (məsələn, **lag_7_days**) indiki zamanla müqayisə edərək proqnozlarını maksimum dərəcədə stabilləşdirir.

**3. Protection Against Overfitting**
Zəif modellər çox vaxt keçmiş məlumatları sadəcə "əzbərləyir" və yeni, görülməmiş məlumatlarda səhv proqnozlar verir. XGBoost isə daxili **Regularization** (L1 və L2) mexanizmləri sayəsində datanı əzbərləmir, onun məntiqini öyrənir. Bu da Kinetix AI-ın real dünyada, yəni **test set** üzərində daha stabil nəticələr göstərməsini təmin edir.

**4. Speed and Resource Efficiency**
Metrodakı **real-time** sıxlıq proqnozları saniyələr içində hesablanmalıdır. XGBoost həm **training**, həm də **inference** sürətinə görə rəqib alqoritmlərdən (məsələn, Random Forest və ya Deep Learning modellərindən) qat-qat sürətlidir. Bu da tətbiqimizin gələcəkdə "real-time" işləməsi üçün bizə böyük üstünlük qazandırır.

**5. Model Interpretability (Feature Importance)**
XGBoost-un bizə təqdim etdiyi ən böyük üstünlük şəffaflıqdır. **Feature Importance** analizi vasitəsilə o, bizə proqnoz verərkən hansı faktorun (məsələn, həftənin günü, yoxsa keçən həftəki sərnişin sayı) daha çox rol oynadığını göstərir. Bu, Kinetix AI-ın sadəcə bir "black box" deyil, hər qərarını izah edə bilən ağıllı bir sistem olmasını təmin edir.

**Yekun Nəticə:** XGBoost seçimi ilə biz Kinetix AI-ın həm **accuracy**, həm sürət, həm də **reliability** göstəricilərini sığortalamış olduq. Bu model Bakı metrosunun mürəkkəb sərnişin xəritəsini rəqəmlərlə oxuyan ən peşəkar analitik mexanizmdir.

(Burada bir şey qeyd edə bilərəm ki, Kinetix AI həmçinin anlıq hava haqqında proqnoza da baxaraq, hansı günlərdə, hansı hava şəraitlərində ictimai nəqliyyatda sıxlığın necə olduğunu predict edə bilsin)

### **Model Training & Intelligence Optimization**

Hazırladığımız bütün datanı **XGBoost** modelinə təqdim etdik və ona Bakı metrosu üçün ən doğru təxminləri necə verəcəyini öyrətdik. Bu proses sadəcə bir düyməyə basmaq deyildi; biz modelin hər bir addımını, yəni **Hyperparameter Configuration** hissəsini xüsusi bir strategiya ilə qurduq:

**1. "Tələsmədən Öyrən" Strategiyası (Learning Rate & Estimators)**

Modelə 1000 fərqli qərar vermə imkanı (**n_estimators=1000**) tanıdıq, lakin ona bir tapşırıq verdik: "Hər bir addımda çox kiçik öyrənmə sürəti ilə hərəkət et (**learning_rate=0.05**)".

- **Niyə?** Çünki süni intellekt tələsdikdə məlumatdakı təsadüfi "küy"ləri (**noise**) əsl trendlərlə qarışdırır. Kiçik addımlarla öyrənmək modelin daha dərindən düşünməsinə və sərnişin axınındakı incə detalları (məsələn, yağışlı bir bazar ertəsi ilə günəşli bir bazar ertəsi arasındakı fərqi) daha yaxşı tutmasına kömək edir.

**2. Dərinlik və Sabitlik Balansı (Complexity Control)**

Hər bir qərar ağacının dərinliyini 6 səviyyə ilə məhdudlaşdırdıq (**max_depth=6**). Eyni zamanda, modelə hər dəfə öyrənərkən məlumatın və xüsusiyyətlərin yalnız 80%-ni görməyə icazə verdik (**subsample** & **colsample_bytree = 0.8**).

- **Nəticə:** Bu, modelin "əzbərçi" olmasının qarşısını alır. O, eyni şeyi təkrar-təkrar görmək əvəzinə, fərqli ssenarilərə baxa-baxa daha dayanıqlı (**Robust**) bir məntiq inkişaf etdirir. Biz buna **Data Science** dilində **Overfitting**dən qorunmaq deyirik.

**3. Early Stopping: Modelin "Pik" Nöqtəsini Tapmaq**

Layihənin ən "ağıllı" addımlarından biri **early_stopping_rounds=50** əmrini vermək oldu.

- **Necə işlədi?** Model öyrənməyə başladı və hər addımda özünü həm **Train Set**, həm də daha öncə görmədiyi **Validation Set** üzərində test etdi. O gördü ki, 116-cı addımdan (**best_iteration=116**) sonra artıq yeni bir şey öyrənmir, əksinə, datanı əzbərləməyə başlayaraq xəta payını artırır.
- **Qərar:** Sistem özü avtomatik olaraq orada dayandı. Bu, bizə həm zaman qənaəti təmin etdi, həm də modelin real həyatda səhv etmə ehtimalını minimuma endirdi.

**4. Uğur Göstəricisi: 2,331 RMSE Nəyi İfadə Edir?**

Nəticədə əldə etdiyimiz **2,331 RMSE (Root Mean Squared Error)** skoru modelimizin "imtahan qiyməti"dir.

- Bu rəqəm bizə deyir ki, Kinetix AI hər bir stansiya üzrə minlərlə sərnişin axınını təxmin edərkən orta hesabla cəmi 2,331 nəfərlik bir sapma ilə işləyir.
- Metronun ümumi sərnişin həcmini nəzərə alsaq, bu, modelin olduqca yüksək dəqiqliklə (**Accuracy**) Bakı metrosunun nəbzini tutduğunu sübut edir.

Model artıq təkcə öyrədilməyib, həm də öz imtahanından uğurla keçib. Cəmi 1934 fərqli ssenari üçün proqnozlar (**Predictions**) artıq hazırdır. Kinetix AI artıq sadə bir proqram deyil, Bakı metrosundakı sərnişin sıxlığını saatlar, günlər öncədən hiss edə bilən peşəkar bir "analitik"dir.

### **Model Evaluation & Performance Analytics**

**1. Evaluation Metrics:** 

Modelin performansını ölçmək üçün istifadə etdiyimiz üç əsas metrik Kinetix AI-ın nə dərəcədə etibarlı olduğunu göstərir:

- **MAE (Mean Absolute Error) - 1,170.431:** Bu göstərici proqnozlaşdırılan sərnişin sayı ilə real sərnişin sayı arasındakı orta mütləq fərqi ifadə edir. Bakı metrosunun bəzi stansiyalarında gündəlik gediş sayının 60,000-dən çox olduğunu nəzərə alsaq, orta hesabla cəmi 1,170 sərnişin yanılma payı modelin **High Precision** (yüksək dəqiqlik) ilə işlədiyini sübut edir.
- **RMSE (Root Mean Squared Error) - 2,331.786:** RMSE böyük xətaları daha ağır cəzalandıran bir metrikdir. RMSE-nin MAE-dən yüksək olması o deməkdir ki, bəzi spesifik saatlarda (məsələn, gözlənilməz pik saatlarda) xəta payı ortalamadan bir qədər yüksək ola bilir. Lakin bu rəqəmin 2,300 ətrafında sabitlənməsi modelin **Overfitting** (əzbərləmə) etmədiyini və **Generalization** qabiliyyətinin yüksək olduğunu göstərir.
- **R² (R-Squared) - 0.9859:** Bu, modelin ən uğurlu göstəricisidir. 0.9859 o deməkdir ki, Bakı metrosundakı sərnişin axınındakı dəyişkənliyin (**Variance**) **98.6%-i** Kinetix AI tərəfindən tam dəqiqliklə izah olunur. Bu, zaman sıraları proqnozlaşdırılmasında çox nadir və yüksək nəticə hesab olunur.

**2. Feature Importance:**

![image.png](attachment:216e7448-66f2-4348-af72-638c57a544ae:image.png)

Bu qrafik bizə modelin sərnişin sayını təxmin edərkən hansı faktorlara üstünlük verdiyini göstərir:

- **Əsas Faktor (rolling_mean_3_days):** Model sərnişin sayını müəyyən edərkən 36.1% payla ən çox son 3 günün ortalama trendinə güvənir. Bu, sistemin qısamüddətli dinamikaya dərhal uyğunlaşdığını göstərir.
- **Mövsümi Təsir (lag_7_days):** 17.3% payla ikinci yerdə keçən həftənin eyni gününün göstəricisi gəlir. Bu, modelin həftəlik **Seasonality** (mövsümilik) trendini çox güclü şəkildə mənimsədiyini sübut edir.
- **Spesifik Stansiya Təsiri (station_28_May):** Maraqlıdır ki, "28 May" stansiyası özlüyündə bir faktor kimi 16.3% əhəmiyyət kəsb edir. Bu, modelin şəhərin mərkəzi tranzit qovşağındakı (**Transfer Hub**) spesifik yüklənmə modelini digər stansiyalardan fərqləndirdiyini göstərir.
- **Local Context:** Bayram günləri (`is_holiday_az`) və akademik mövsümün (`is_academic_season`) də modelin qərarlarına birbaşa təsir etməsi Kinetix AI-ın yerli reallıqları nəzərə alan bir **Context-Aware** model olduğunu təsdiqləyir.

**3. Actual vs. Predicted:**

![image.png](attachment:8c86cb20-5277-4a2b-9364-ec7ad486b731:image.png)

Bu qrafik modelin real həyatdakı sınağını vizual olaraq əks etdirir:

- **Xronoloji Uyğunluq:** Göy (Actual) və narıncı (Predicted) xətlər demək olar ki, sinxron hərəkət edir. Model sərnişin sayındakı kəskin artışları (peaks) və azalma nöqtələrini (troughs) tam vaxtında tutur.
- **Trend Analizi:** Diqqət etsəniz, model hətta kəskin sıçrayışların baş verdiyi günlərdə belə trayektoriyanı itirmir. Bu, modelin **Time-Series Alignment** (zaman sırası düzləndirilməsi) məsələsini mükəmməl həll etdiyini və real-time rejimdə işləməyə hazır olduğunu göstərir.

**4. Residual Analysis: Xətaların Paylanması**

![image.png](attachment:aa4cdba4-cda7-4f38-8642-2f7b1c328f96:image.png)

Xətaların analizi (**Residuals Analysis**) modelin "sağlamlığını" yoxlamaq üçün ən texniki üsuldur:

- **Normal Distribution:** Xətaların paylanması qrafiki tam bir "zəngvari" (**Bell Curve**) forma yaradır və mərkəzi "0" nöqtəsindədir.
- **Unbiased Prediction:** Xətaların mərkəzinin 0-da olması modelin **Unbiased** (qərəzsiz) olduğunu sübut edir. Yəni model nəticələri sistematik olaraq nə yuxarı, nə də aşağı yuvarlaqlaşdırmır; səhvlər tamamilə təsadüfidir və minimaldır.
- **Error Density:** Xətaların böyük əksəriyyətinin -2500 ilə +2500 aralığında sıxlaşması (minlərlə sərnişin axını daxilində) sistemin stabil işlədiyini təsdiq edir.

Bu texniki analizlər sübut edir ki, Kinetix AI artıq sadəcə bir proqnozlaşdırma aləti deyil, Bakı metrosunun sərnişin axınını **98.6%** dəqiqliklə anlayan və təsvir edən güclü bir **Analitik İnfrastruktur**dur.

Növbəti mərhələdə bu proqnozları stansiya tutumları (**Station Capacity**) ilə kəsişdirərək, sərnişinlər üçün real-zamanlı "Sıxlıq Xəritəsi" (Density Heatmap) və "Komfort Səviyyəsi" (Comfort Band) hesabatlarını hazırlamağa tam hazırıq.

### **Exploratory Data Analysis (EDA)**

**1. Məlumatların Strukturlaşdırılması (Canonicalization)**

Kodun ilk hissəsində biz fərqli mənbələrdən gələ biləcək sütun adlarını (məsələn: "tarix", "date", "stansiya", "station") vahid bir formata gətirdik.

- **Nə etdik?** `column_map` vasitəsilə sütunları **date**, **station_name** və **passenger_count** olaraq standartlaşdırdıq.
- **Niyə etdik?** Kodun universal olması üçün. Datasetin sütun adları dəyişsə belə, bu məntiq onları tanıyır və analizi davam etdirir. Həmçinin, `hour_of_day`, `day_name` və `month` kimi yeni **temporal** sütunlar yaradaraq datanı daha detallı analiz üçün hazırladıq.

**2. Basic Data Profiling (Statistik Xülasə)**

Çıxışdakı statistik rəqəmlər bizə Bakı metrosunun ümumi mənzərəsini göstərir:

- **Mean (23,014):** Orta hesabla hər stansiyadan gündəlik 23 min sərnişin keçir.
- **Max (77,392):** Ən pik nöqtədə bir stansiyadan gün ərzində 77 mindən çox sərnişin daxil olub.
- **Outlier Check:** **IQR (Interquartile Range)** metodu ilə anomal sıçrayışları yoxladıq. Nəticədə 109 minlik limitdən yuxarı heç bir kəskin anomal sıçrayış tapılmadı, bu da datanın real və etibarlı olduğunu göstərir.

**Total Daily Passenger Flow Over Time**

![image.png](attachment:abf1985f-d95a-42ed-b088-9628842543f9:image.png)

Bu qrafik Bakı metrosunun bir il ərzindəki "nəbzini" göstərir.

- **Göy xətt (Daily Total):** Gündəlik sərnişin sayının dəyişməsidir. Göründüyü kimi, həftəlik kəskin dalğalanmalar var (iş günləri yüksək, istirahət günləri aşağı).
- **Qırmızı xətt (7-Day Rolling Average):** Bu xətt gündəlik küyü (**noise**) təmizləyərək bizə əsas trendi göstərir.
- **Müşahidə:** İyun-Avqust aylarında (yay mövsümü) sərnişin axınında ciddi bir düşüş müşahidə olunur. Sentyabrın ortalarından etibarən isə təhsil mövsümünün açılması ilə kəskin artım başlayır. Bu, modelimiz üçün **is_academic_season** faktorunun nə qədər önəmli olduğunu təsdiqləyir.

**Baku Metro Seasonality Analysis**

![image.png](attachment:d9353907-ae4a-4095-bcdc-d24516e14e75:image.png)

Bu iki qrafik sərnişin axınının həftəlik və aylıq ritmini izah edir.

- **Sol tərəf (Day of Week):** Ən çox yüklənmə Çərşənbə axşamı (Tuesday) müşahidə olunur. Bazar günü (Sunday) isə sərnişin sayı kəskin şəkildə azalır (təxminən 25 mindən 15 minə düşür).
- **Sağ tərəf (Month):** Oktyabr və Dekabr ayları ən gərgin aylardır. Avqust ayı isə ilin ən sakin ayıdır. Bu məlumat modelin **seasonality** trendlərini öyrənməsi üçün təməl bazadır.

**Station Traffic Comparison**

![image.png](attachment:91909f5e-5197-4dc2-9461-789d1fe029bb:image.png)

Bu qrafik stansiyalar arasındakı "güc nisbətini" göstərir.

- **Top 10 Busiest:** **Koroğlu**, **28 May** və **20 Yanvar** stansiyaları şəhərin əsas yüklənmə mərkəzləridir. Koroğlu stansiyası gündəlik 55 min+ sərnişinlə liderdir.
- **Bottom 10 Least Busy:** **Memar Əcəmi-2** və **Bakmil** stansiyaları ən az giriş edilən stansiyalardır.
- **Məntiq:** Bu fərqlilik modelə hər stansiya üçün fərdi çəki dərəcəsi (**weight**) təyin etməyə imkan verir.

**Kinetix Heatmap**

![image.png](attachment:142e107b-77e2-45bf-932c-420fda978075:image.png)

Bu istilik xəritəsi (Heatmap) günün saatları ilə həftənin günləri arasındakı kəsişməni göstərir.

- **Rəng intensivliyi:** Tünd qırmızı rənglər ən yüksək sıxlığı, sarı rənglər isə seyrəkliyi ifadə edir.
- **Müşahidə:** İş günləri (Bazar ertəsi - Cümə) intensivlik eyni səviyyədə yüksəkdir. Şənbə günü rəng bir az açılır, Bazar günü isə tamamilə sarı rəngə (aşağı sıxlığa) keçir. Bu, modelin həftəsonu faktorunu necə interpretasiya edəcəyini vizuallaşdırır.

**Yekun Nəticə**

Bu EDA mərhələsi bizə sübut etdi ki, Bakı metrosunun datası həm **temporal** (zamanla bağlı), həm də **spatial** (stansiya ilə bağlı) çox güclü qanunauyğunluqlara malikdir. Biz artıq bilirik ki:

1. Yay və təhsil mövsümü arasındakı fərq kəskindir.
2. Bazar günü tamamilə fərqli bir rejimdir.
3. Koroğlu və 28 May kimi stansiyalar model üçün prioritetdir.

İndi biz tam əminliklə növbəti mərhələyə — **Feature Engineering** və **Model Training** hissəsinə keçə bilərik, çünki datanın bizə nə demək istədiyini artıq vizual olaraq başa düşmüşük.

![image.png](attachment:c9aec38b-e0b3-413f-a760-fb550ecb510d:image.png)

**1. Monthly Usage Trend (Yuxarı Sol: Bar Chart)**

Bu qrafik metronun il ərzindəki makro-ritmini göstərir.

- **Nəyi göstərir:** Hər ay üzrə cəmi sərnişin sayını.
- **Analiz:** Qrafikdə **"Summer Drop"** (Yay düşüşü) və **"Winter Peaks"** (Qış zirvələri) aydın görünür. İyun, İyul və Avqust aylarında sərnişin sayı minimuma düşür. Bu, dərslərin bitməsi, insanların məzuniyyətə getməsi və şəhərin boşalması ilə izah olunur.
- **Nəticə:** Sentyabr ayından etibarən başlayan kəskin artım **Academic Season** (təhsil mövsümü) faktorunun model üçün nə dərəcədə həlledici olduğunu sübut edir.

**2. Day of Week Usage (Yuxarı Sağ: Box Plot)**

Bu qrafik həftənin günləri üzrə sərnişin paylanmasını və məlumatın **variance** (dəyişkənlik) dərəcəsini göstərir.

- **Nəyi göstərir:** Hər gün üzrə sərnişin sayının medianını, kvartillərini və kənar sapmalarını (**outliers**).
- **Analiz:** İş günləri (Bazar ertəsi - Cuma) bir-birinə çox yaxın və yüksək yüklənmə ilə xarakterizə olunur. Lakin Şənbə və xüsusilə Bazar günü sərnişin sayı kəskin aşağı düşür.
- **Nəticə:** Qrafikdəki "qutuların" (boxes) ölçüsü bizə göstərir ki, iş günlərində sərnişin axını daha stabil və proqnozlaşdırıla biləndir, həftəsonları isə bir az daha fərqli (müxtəlif) ola bilir.

**3. Average Passenger Density: Month vs. Day of Week (Aşağı Sol: Heatmap)**

Bu layihənin ən granular (detallı) vizuallaşdırmasıdır.

- **Nəyi göstərir:** Ayların və həftənin günlərinin kəsişməsində orta sərnişin sayını.
- **Analiz:** Rəng nə qədər tünd qırmızıdırsa, sıxlıq bir o qədər yüksəkdir. Göründüyü kimi, Dekabr və Noyabr aylarının iş günləri (təqvim üzrə 11 və 12-ci aylar) sistemin ən pik nöqtələridir (təxminən 30,000+ sərnişin/stansiya).
- **Nəticə:** Bu **Heatmap** bizə imkan verir ki, ilin hansı spesifik günlərində metronun ən yüksək təzyiq altında olduğunu dəqiq müəyyən edək. Məsələn, Avqust ayının Bazar günü ilin ən sakin vaxtıdır.

**4. Anomaly Detection: Total Daily Passengers (Aşağı Sağ: Time-Series Line Chart)**

Bu qrafik sistemin "sağlamlıq yoxlamasıdır".

- **Nəyi göstərir:** Gündəlik sərnişin sayını, 7 günlük sürüşən ortalamanı (**7-Day Rolling Avg**) və Z-score əsaslı anomaliyaları.
- **Analiz:** Göy xətt kəskin dalğalanmaları, qırmızı qırıq xətt isə ümumi trendi göstərir. Qırmızı nöqtələr isə **Z-score >= 2.5** olan günləri, yəni statistik olaraq gözlənilməz, qeyri-adi dərəcədə yüksək və ya aşağı sərnişin axını olan günləri işarələyir.
- **Nəticə:** Bu anomaliyalar adətən fövqəladə hallar, kütləvi tədbirlər və ya bayram tətilləri ilə üst-üstə düşür. Bu, Kinetix AI modelinə öyrədir ki, hansı günləri "normal" qəbul etməli, hansılarını isə "istisna" (special case) kimi dəyərləndirməlidir.

**Ümumi Yekun**

Bu Dashboard sübut edir ki, Bakı metrosunun sərnişin axını təsadüfi deyil, çox ciddi bir **Seasonal Pattern**-ə (mövsümi qanunauyğunluğa) tabedir.

1. **Macro səviyyədə:** Yay və Qış mövsümləri arasındakı fərq kəskindir.
2. **Micro səviyyədə:** Həftə içi və Həftə sonu rejimləri tamamilə fərqlidir.
3. **Dəqiqlik:** Anomaliyaların (qırmızı nöqtələr) sayının az olması datanın kifayət qədər təmiz və model qurmaq üçün ideal olduğunu göstərir.

Bu analizlər artıq modelimizin "nəyi proqnozlaşdırdığını" tam başa düşməyimizə şərait yaradır.

### **Density Metrics & Crowd Behavioral Analytics**

Bu fazada biz sadəcə sərnişin sayını deyil, həmin sayın stansiya mühiti üçün yaratdığı **Density** (sıxlıq) yükünü hesabladıq. Sizin təyin etdiyiniz 5,000 nəfərlik **Station Capacity** əsasında aparılan analizlər Bakı metrosundakı gərginliyi rəqəmlərlə sübut edir.

### 1. Density Summary: Statistik Təhlil

Hesablamalar göstərir ki, stansiyalarımız orta hesabla öz tutumlarını dəfələrlə üstələyir:

- **Mean Density (4.60):** Orta hesabla stansiyalar ideal tutumlarından **4.6 dəfə** daha artıq yüklənir.
- **Max Density (15.47):** Ən pik nöqtələrdə stansiyalar öz tutumlarını **15 qatdan çox** aşır. Bu rəqəm artıq **Critical Overcrowding** vəziyyətini ifadə edir.
- **High-Density Event Rate (~83.7%):** Ümumi ölçümlərin 8,247-si (9,855-dən) **High Density** (Sıxlıq > 0.75) zonasına düşür. Bu o deməkdir ki, günün böyük bir hissəsində stansiyalarımız "Qırmızı Zona"da fəaliyyət göstərir.

**High-Density Event Rate Heatmaps**

![image.png](attachment:e0cae1d6-2ea2-4009-bc8f-e7c3f41ec97c:image.png)

Bu qrafik bizə sıxlığın zamana (ay və saat) görə necə paylandığını **faceted heatmap** formatında göstərir.

- **Nəyi göstərir:** Həftənin hər günü üçün aylıq (Y oxu) və saatlıq (X oxu) sıxlıq dərəcəsini.
- **Texniki Analiz:** * Heatmap-lərdəki tünd qırmızı sütunlar (xüsusilə iş günlərinin başlanğıc saatlarında) 100% **High-Density** ehtimalını göstərir.
    - Diqqət çəkən məqam odur ki, sıxlıq təkcə müəyyən saatlarda deyil, bütün aylar boyu stabil bir **pattern** (qanunauyğunluq) izləyir.
    - Həftəsonları (Sat, Sun) qrafiklərdəki rəng intensivliyinin dəyişməməsi, Bakı metrosunun hətta istirahət günlərində belə yüksək yük altında işlədiyini, lakin iş günlərinə nisbətən bu yüklənmənin daha "idarəolunan" (lower variance) olduğunu göstərir.

**Average Consecutive High-Density Duration**

![image.png](attachment:94710578-dd8e-4620-9f28-b5dd73b8fbdd:image.png)

Bu bar-chart stansiyaların "nə qədər müddət dalbadal" qırmızı zonada qaldığını ölçür. Bu, **Operational Logistics** üçün ən vacib göstəricidir.

- **Nəyi göstərir:** Bir stansiya yüksək sıxlıq zonasına girdikdən sonra həmin vəziyyətin orta hesabla neçə saat davam etdiyini.
- **Texniki Analiz:** * **20 Yanvar**, **28 May**, **Koroğlu** və **Gənclik** kimi mərkəzi stansiyalarda bu müddət **1.0 saata** bərabərdir. Bu o deməkdir ki, bu stansiyalar sıxlıq zonasına daxil oldusa, bu vəziyyət ən azı bir tam saat boyunca stabil qalır.
    - **Bakmil** və **Memar Əcəmi-2** kimi stansiyalarda isə bu müddət kəskin aşağıdır (0.2 - 0.4 saat). Bu, həmin stansiyaların sərnişin axınının daha çox **transient** (keçici) xarakter daşıdığını və sıxlığın tez bir zamanda seyrəldiyini göstərir.
- **Nəticə:** Bu məlumat əsasında stansiya heyətinin (polis və dispetçerlər) müdaxilə müddəti optimallaşdırıla bilər.

Analizimiz sübut etdi ki, Kinetix AI sadəcə "neçə nəfər gələcək?" sualına deyil, həm də **"bu sərnişin yükü nə qədər müddət davam edəcək?"** sualına cavab verir.

1. **High-Density Persistence:** Əsas stansiyalarda sıxlıq uzunmüddətlidir, bu da qatarların intervalının həmin saatlarda minimal saxlanılmasını tələb edir.
2. **Capacity Re-evaluation:** Mövcud 5,000 nəfərlik limit Bakı metrosu üçün çox kiçikdir (çünki mean density 4.6-dır). Bu o deməkdir ki, stansiyalarımız texniki olaraq daim öz limitində işləyir.

Kinetix AI artıq bir **Decision Support System** (qərar dəstək sistemi) kimi formalaşıb.

**Top 10 Busiest Stations by Mean Density per Month**

![image.png](attachment:e8408227-37e3-47c8-b62d-381993f96e1a:image.png)

Bu qrafiklər ilin hər bir ayı üçün ən çox yüklənən 10 stansiyanı **Comfort Band** rəngləri ilə təsnif edir.

- **Nəyi göstərir:** Hər ay üçün stansiyaların orta sıxlıq dərəcəsini.
- **Analiz:** Bütün aylarda bütün sütunların **Qırmızı (Red > 75%)** olması, seçilmiş ən sıx 10 stansiyanın heç vaxt "Yaşıl" və ya "Sarı" zonaya düşmədiyini göstərir.
- **Pik Ayları:** Dekabr (Month 12) və Yanvar (Month 1) aylarında Koroğlu və 28 May stansiyalarında orta sıxlıq 12.0-ni keçir. Bu, tutumun **1200%** aşılması deməkdir.
