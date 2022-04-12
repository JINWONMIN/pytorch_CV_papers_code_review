<div align="center">

# U-Net 논문 구현

</div>

:bulb: **U-Net 구현 순서**

---
1. 데이터 다운
2. UNet 구현
3. Data Loader & Transform
4. Network Training & Test

<br>

:bulb: **Code framework**

---
1. train.py
    - Training & Test

2. datast.py
    - Dataset & Transform

3. model.py
    - Network models
 
4. util.py
    - Network's Save & Load

<br>

:bulb: **Data**

---
1. **input**

![화면 캡처 2022-04-12 105135](https://user-images.githubusercontent.com/94345086/162863063-cbfc186c-853f-4e62-8b9c-2e66e9be2951.png)

2. **label(정답)**

![화면 캡처 2022-04-12 105153](https://user-images.githubusercontent.com/94345086/162863094-049488b8-517c-4be9-be13-3c364b8bb8cf.png)

3. **output(모델의 출력 값)**

![화면 캡처 2022-04-12 105218](https://user-images.githubusercontent.com/94345086/162863114-99dc40ea-f7b6-42a9-8817-1f6b92b64506.png)

<br>
<br>
<br>
<br>

#### Reference

---
- <https://www.youtube.com/watch?v=aNAaxy8n-AQ&list=PLqtXapA2WDqbE6ghoiEJIrmEnndQ7ouys&index=2>
