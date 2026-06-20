# SRAM Minimum Capacity Calculation Methodology (Fixed Slicing)

To maximize the profit score, we must size our SRAM banks to precisely fit the mathematical bounds of our chosen "Fixed Slicing" method (Layer 0 Vertical, Layers 1+ Horizontal).

## 1. The Core Formulas

For any given model assigned $C$ cores with architecture layers $[L_0, L_1, L_2, \dots, L_n]$:

* **Broadcast Input:** $L_0$
* **Layer Shards (Chunks):** The size of data held locally per core for layer $k$ is $S_k = \lceil L_k / C 
ceil$. The **Maximum Shard** is $\max(S_1, S_2, \dots, S_n)$.
* **Partial Sums:** During a row-split layer (Layers 1+), the Array module computes a full unreduced partial sum before it is chunked. The **Maximum Partial Sum** is $\max(L_2, L_3, \dots, L_n)$.

**How these map to hardware banks:**
* **Bank 0 (Array Input):** $\max(	ext{Broadcast}, 	ext{Maximum Shard})$
* **Bank 1 (Array Output):** $\max(	ext{Maximum Shard}, 	ext{Maximum Partial Sum})$
* **Banks 2, 3, 4 (Vector Units):** $	ext{Maximum Shard}$ *(Because Vector units only compute the post-reduce-scatter chunks!)*

---

## 2. Step-by-Step Model Calculations

### Model 0 (Cores: 6)
* **Architecture:** `440 · 332 · 3475 · 768 · 3766 · 729 · 294 · 912 · 9`
* **Broadcast Size:** `440`
* **Shards ($\lceil L_k / 6 \rceil$):** `56, 580, 128, 628, 122, 49, 152, 2` $\rightarrow$ **Max Shard:** `628`
* **Partial Sums:** `3475, 768, 3766, 729, 294, 912, 9` $\rightarrow$ **Max Partial Sum:** `3766`
* **Derived Banks:**
  * Bank 0 (Array In) = $\max(440, 628)$ = **628**
  * Bank 1 (Array Out) = $\max(628, 3766)$ = **3766**
  * Vector Banks = **628**

### Model 1 (Cores: 2)
* **Architecture:** `297 · 2975 · 793 · 372 · 13`
* **Broadcast Size:** `297`
* **Shards ($\lceil L_k / 2 \rceil$):** `1488, 397, 186, 7` $\rightarrow$ **Max Shard:** `1488`
* **Partial Sums:** `793, 372, 13` $\rightarrow$ **Max Partial Sum:** `793`
* **Derived Banks:**
  * Bank 0 (Array In) = $\max(297, 1488)$ = **1488**
  * Bank 1 (Array Out) = $\max(1488, 793)$ = **1488**
  * Vector Banks = **1488**

### Model 2 (Cores: 2)
* **Architecture:** `93 · 2187 · 1098 · 1327 · 384 · 10`
* **Broadcast Size:** `93`
* **Shards ($\lceil L_k / 2 \rceil$):** `1094, 549, 664, 192, 5` $\rightarrow$ **Max Shard:** `1094`
* **Partial Sums:** `1098, 1327, 384, 10` $\rightarrow$ **Max Partial Sum:** `1327`
* **Derived Banks:**
  * Bank 0 (Array In) = $\max(93, 1094)$ = **1094**
  * Bank 1 (Array Out) = $\max(1094, 1327)$ = **1327**
  * Vector Banks = **1094**

### Model 3 (Cores: 15)
* **Architecture:** `350 · 3619 · 2119 · 2830 · 498 · 3402 · 2818 · 11`
* **Broadcast Size:** `350`
* **Shards ($\lceil L_k / 15 \rceil$):** `242, 142, 189, 34, 227, 188, 1` $\rightarrow$ **Max Shard:** `242`
* **Partial Sums:** `2119, 2830, 498, 3402, 2818, 11` $\rightarrow$ **Max Partial Sum:** `3402`
* **Derived Banks:**
  * Bank 0 (Array In) = $\max(350, 242)$ = **350**
  * Bank 1 (Array Out) = $\max(242, 3402)$ = **3402**
  * Vector Banks = **242**

### Model 4 (Cores: 8)
* **Architecture:** `435 · 1547 · 2669 · 1492 · 3703 · 7`
* **Broadcast Size:** `435`
* **Shards ($\lceil L_k / 8 \rceil$):** `194, 334, 187, 463, 1` $\rightarrow$ **Max Shard:** `463`
* **Partial Sums:** `2669, 1492, 3703, 7` $\rightarrow$ **Max Partial Sum:** `3703`
* **Derived Banks:**
  * Bank 0 (Array In) = $\max(435, 463)$ = **463**
  * Bank 1 (Array Out) = $\max(463, 3703)$ = **3703**
  * Vector Banks = **463**

### Model 5 (Cores: 6)
* **Architecture:** `66 · 1454 · 3586 · 699 · 4058 · 10`
* **Broadcast Size:** `66`
* **Shards ($\lceil L_k / 6 \rceil$):** `243, 598, 117, 677, 2` $\rightarrow$ **Max Shard:** `677`
* **Partial Sums:** `3586, 699, 4058, 10` $\rightarrow$ **Max Partial Sum:** `4058`
* **Derived Banks:**
  * Bank 0 (Array In) = $\max(66, 677)$ = **677**
  * Bank 1 (Array Out) = $\max(677, 4058)$ = **4058**
  * Vector Banks = **677**

### Model 6 (Cores: 7)
* **Architecture:** `184 · 1949 · 3179 · 2130 · 19`
* **Broadcast Size:** `184`
* **Shards ($\lceil L_k / 7 \rceil$):** `279, 455, 305, 3` $\rightarrow$ **Max Shard:** `455`
* **Partial Sums:** `3179, 2130, 19` $\rightarrow$ **Max Partial Sum:** `3179`
* **Derived Banks:**
  * Bank 0 (Array In) = $\max(184, 455)$ = **455**
  * Bank 1 (Array Out) = $\max(455, 3179)$ = **3179**
  * Vector Banks = **455**

### Model 7 (Cores: 9)
* **Architecture:** `127 · 2652 · 1099 · 1434 · 3623 · 1655 · 12`
* **Broadcast Size:** `127`
* **Shards ($\lceil L_k / 9 \rceil$):** `295, 123, 160, 403, 184, 2` $\rightarrow$ **Max Shard:** `403`
* **Partial Sums:** `1099, 1434, 3623, 1655, 12` $\rightarrow$ **Max Partial Sum:** `3623`
* **Derived Banks:**
  * Bank 0 (Array In) = $\max(127, 403)$ = **403**
  * Bank 1 (Array Out) = $\max(403, 3623)$ = **3623**
  * Vector Banks = **403**

### Model 8 (Cores: 8)
* **Architecture:** `433 · 1415 · 2767 · 2962 · 729 · 16`
* **Broadcast Size:** `433`
* **Shards ($\lceil L_k / 8 \rceil$):** `177, 346, 371, 92, 2` $\rightarrow$ **Max Shard:** `371`
* **Partial Sums:** `2767, 2962, 729, 16` $\rightarrow$ **Max Partial Sum:** `2962`
* **Derived Banks:**
  * Bank 0 (Array In) = $\max(433, 371)$ = **433**
  * Bank 1 (Array Out) = $\max(371, 2962)$ = **2962**
  * Vector Banks = **371**

### Model 9 (Cores: 1)
* **Architecture:** `374 · 825 · 15`
* **Broadcast Size:** `374`
* **Shards ($\lceil L_k / 1 \rceil$):** `825, 15` $\rightarrow$ **Max Shard:** `825`
* **Partial Sums:** `15` $\rightarrow$ **Max Partial Sum:** `15`
* **Derived Banks:**
  * Bank 0 (Array In) = $\max(374, 825)$ = **825**
  * Bank 1 (Array Out) = $\max(825, 15)$ = **825**
  * Vector Banks = **825**

---

## 3. Global Configuration Summary Table

This table aggregates the calculations above. Because the `student_hw_GXX.h` file applies a single configuration to the entire 64-core array, you must select the **absolute global maximum** in each column to prevent out-of-bounds errors on any core.

| Model | Cores | Architecture | Min Array In (Bank 0) | Min Array Out (Bank 1) | Min Vector (Banks 2,3,4) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **0** | 6 | 440 · 332 · 3475 · 768 · 3766 · 729 · 294 · 912 · 9 | 628 | 3766 | 628 |
| **1** | 2 | 297 · 2975 · 793 · 372 · 13 | 1488 | 1488 | 1488 |
| **2** | 2 | 93 · 2187 · 1098 · 1327 · 384 · 10 | 1094 | 1327 | 1094 |
| **3** | 15 | 350 · 3619 · 2119 · 2830 · 498 · 3402 · 2818 · 11 | 350 | 3402 | 242 |
| **4** | 8 | 435 · 1547 · 2669 · 1492 · 3703 · 7 | 463 | 3703 | 463 |
| **5** | 6 | 66 · 1454 · 3586 · 699 · 4058 · 10 | 677 | 4058 | 677 |
| **6** | 7 | 184 · 1949 · 3179 · 2130 · 19 | 455 | 3179 | 455 |
| **7** | 9 | 127 · 2652 · 1099 · 1434 · 3623 · 1655 · 12 | 403 | 3623 | 403 |
| **8** | 8 | 433 · 1415 · 2767 · 2962 · 729 · 16 | 433 | 2962 | 371 |
| **9** | 1 | 374 · 825 · 15 | 825 | 825 | 825 |

**Optimal Hardware Configuration:**
* `BUF_ARRAY_IN_SIZE` = **1488** (Dictated by Model 1)
* `BUF_ARRAY_OUT_SIZE` = **4058** (Dictated by Model 5)
* `BUF_VEC_IN_SIZE` = **1488** (Dictated by Model 1)
* `BUF_VEC_W_SIZE` = **1488** (Dictated by Model 1)
* `BUF_VEC_OUT_SIZE` = **1488** (Dictated by Model 1)

| Y \ X | 0              | 1              | 2              | 3              | 4              | 5              | 6              | 7              |
|-------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|
| 0     | Core 0 [0,0]   | Core 1 [1,0]   | Core 2 [2,0]   | Core 3 [3,0]   | Core 4 [4,0]   | Core 5 [5,0]   | Core 6 [6,0]   | Core 7 [7,0]   |
| 1     | Core 8 [0,1]   | Core 9 [1,1]   | Core 10 [2,1]  | Core 11 [3,1]  | Core 12 [4,1]  | Core 13 [5,1]  | Core 14 [6,1]  | Core 15 [7,1]  |
| 2     | Core 16 [0,2]  | Core 17 [1,2]  | Core 18 [2,2]  | Core 19 [3,2]  | Core 20 [4,2]  | Core 21 [5,2]  | Core 22 [6,2]  | Core 23 [7,2]  |
| 3     | Core 24 [0,3]  | Core 25 [1,3]  | Core 26 [2,3]  | Core 27 [3,3]  | Core 28 [4,3]  | Core 29 [5,3]  | Core 30 [6,3]  | Core 31 [7,3]  |
| 4     | Core 32 [0,4]  | Core 33 [1,4]  | Core 34 [2,4]  | Core 35 [3,4]  | Core 36 [4,4]  | Core 37 [5,4]  | Core 38 [6,4]  | Core 39 [7,4]  |
| 5     | Core 40 [0,5]  | Core 41 [1,5]  | Core 42 [2,5]  | Core 43 [3,5]  | Core 44 [4,5]  | Core 45 [5,5]  | Core 46 [6,5]  | Core 47 [7,5]  |
| 6     | Core 48 [0,6]  | Core 49 [1,6]  | Core 50 [2,6]  | Core 51 [3,6]  | Core 52 [4,6]  | Core 53 [5,6]  | Core 54 [6,6]  | Core 55 [7,6]  |
| 7     | Core 56 [0,7]  | Core 57 [1,7]  | Core 58 [2,7]  | Core 59 [3,7]  | Core 60 [4,7]  | Core 61 [5,7]  | Core 62 [6,7]  | Core 63 [7,7]  |
