### Breast Ultrasound Images: Segmentation with U-Net and Classification with CNN
----

**python implementation**

```python
Version: 0.0.1  
Author : Md. Rezwanul Haque,
         Anik Ghosh
```
### **Related resources**:


**LOCAL ENVIRONMENT**  
```python
OS          : Ubuntu 22.04 LTS       
Memory      : 8.0 GiB 
Processor   : Intel® Core™ i5-5200U CPU @ 2.20GHz × 4    
Graphics    : AMD® Radeon r5 m255 / Mesa Intel® HD Graphics 5500 (BDW GT2)  
Gnome       : 42.1 
```

**python requirements**
* **pip requirements**: ```pip install -r requirements.txt``` 
> Its better to use a virtual environment 
OR use conda-
* **conda**: use environment.yml: ```conda env create -f environment.yml```


# dataset descriptions
* **[QAMEBI: Breast Ultrasound Images Database](https://qamebi.com/breast-ultrasound-images-database/?fbclid=IwAR1Gu995G1JZQRa59fzTC2dRkfNSXMbpuTFRhXoIdnLFpZwLNWmP8MngQYg)**

---


<!-- create a dataset as follows:
 
```
dataset_folder
    |-raw_videos
        |-sample_1.mp4
        |-2.mp4
        |-sample-3.mp4
        |-...........
        |-...........

    |-data.csv

```

```
* data.csv colums:

    * SL              : Serial No.
    * Patient's ID    : ID of Subjects (like, 0001, 0901, ...)
    * Name            : Name of subjects (xyz, abcd, ...)
    * Age             : Age of subjects (12,69, ...)
    * Sex(M/F)        : Male or Female (M/F)
    * File_ext(*.mp4) : video file extension (.mp4)
    * Hb (g/dL)       : Hemoglobin concentration
    * Gl (mmol/L)     : Glucose concentration
    * Cr (ml/dl)      : Creatinine concentration
```

# Execution
- ```conda activate my_env```
- ```cd scripts```
- run: ```./server.sh```


<!-- **LaTex Utils Install**

```sudo apt install texlive-latex-base```

```sudo apt-get install texlive-latex-extra```

# Execution
- ```conda activate your_env```
- ```cd scripts```
- run: ```./server.sh```


- use **debug.ipynb** for visualization -->

---
