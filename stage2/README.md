## 📁 Data Preparation

Copy the output directory from `stage1` into the `../emc_data` folder. Currently, results for two datasets are included. More datasets can be downloaded from the provided [link](https://www.alipan.com/s/fUsPqpuSHML).

## 🚀 Reproducing Results in Table B.1

Run the corresponding *.sh scripts to reproduce the results reported in **Table B.1**.

### Linear Regression

```bash
bash MVSA.sh
bash CrisisMMD.sh
bash N24News.sh
bash food101.sh
```

### Random Forest

```bash
bash MVSA_rf.sh
bash CrisisMMD_rf.sh
bash N24News_rf.sh
bash food101_rf.sh
```

### Support Vector Regression (SVR)

```bash
bash MVSA_svr.sh
bash CrisisMMD_svr.sh
bash N24News_svr.sh
bash food101_svr.sh
```

### EMLP

```bash
bash MVSA3.sh
bash CrisisMMD2.sh
bash N24News2.sh
bash food1013.sh
```

## 📊 Results (Table B.1)


| Method | MVSA σ=0,p=0 | MVSA σ=5,p=0.5 | MVSA σ=10,p=1 | CrisisMMD σ=0,p=0 | CrisisMMD σ=5,p=0.5 | CrisisMMD σ=10,p=1 |
|--------|---------------|----------------|----------------|--------------------|----------------------|---------------------|
| LR     | 79.46±1.54    | 73.99±1.58     | 67.13±1.16     | 87.67±0.46         | 79.77±1.04           | 73.73±0.65          |
| RF     | 79.21±1.25    | 73.76±1.54     | **67.65±1.17** | 87.55±0.42         | 79.96±0.90           | **74.89±0.53**      |
| SVR    | **79.63±1.47**| 73.66±1.76     | 67.57±1.20     | 87.50±0.50         | 79.53±0.87           | 72.94±0.44          |
| EMLP   | 79.50±1.51    | **74.32±1.78** | 67.50±1.49     | **87.71±0.40**     | **80.26±0.91**       | 74.83±0.48          |


| Method | N24News σ=0,p=0 | N24News σ=5,p=0.5 | N24News σ=10,p=1 | Food101 σ=0,p=0 | Food101 σ=5,p=0.5 | Food101 σ=10,p=1 |
|--------|------------------|-------------------|------------------|------------------|-------------------|------------------|
| LR     | 79.89±0.21       | 66.76±0.69        | 55.28±1.57       | 93.86±0.11       | 76.87±0.33        | 64.24±0.21       |
| RF     | 79.79±0.24       | 67.42±0.69        | 58.06±0.33       | 93.86±0.08       | 77.03±0.29        | 64.26±0.22       |
| SVR    | 79.74±0.31       | 66.96±0.69        | 57.16±0.35       | 93.83±0.08       | 76.65±0.26        | 64.11±0.22       |
| EMLP   | **79.90±0.23**   | **68.61±0.44**    | **58.17±0.38**   | **94.02±0.10**   | **77.50±0.27**    | **64.27±0.21**   |