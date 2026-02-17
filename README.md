# NeRF 相機位姿 3D 視覺化 & 自動化校驗工具

互動式 3D 檢視器，用於視覺化 NeRF 資料集的相機位姿分布，並提供基於**極線幾何 (Epipolar Geometry)** 的自動化校驗功能，偵測可能有誤的相機位姿。

![3D Viewer](https://img.shields.io/badge/3D-Three.js-blue) ![Backend](https://img.shields.io/badge/Backend-Flask-green) ![CV](https://img.shields.io/badge/CV-OpenCV-red)

## 功能

- **3D 視覺化** — 在互動式 Three.js 場景中檢視所有相機的位置和朝向
- **圖片預覽** — 點擊任一相機即可預覽對應圖片及完整位姿矩陣
- **Split 篩選** — 依 train / val / test 分組過濾顯示
- **極線校驗** — 一鍵自動化檢測相機位姿精度，結果以顏色編碼呈現
- **詳細報告** — 點擊相機可查看極線誤差值、評等、鄰近配對明細

## 快速開始

### 1. 下載資料集

資料集來自 [NeRF 官方專案頁面](https://www.matthewtancik.com/nerf)，可從以下 Google Drive 下載：

> **下載連結：** [https://drive.google.com/drive/folders/1cK3UDIJqKAAm7zyrxRYVFJ0BRMgrwhh4](https://drive.google.com/drive/folders/1cK3UDIJqKAAm7zyrxRYVFJ0BRMgrwhh4)

下載後解壓到專案根目錄的 `dataset/` 資料夾，結構如下：

```
dataset/
├── nerf_example_data/nerf_synthetic/   # Synthetic 合成資料集
│   ├── lego/
│   │   ├── transforms_train.json       # 包含 camera_angle_x + 每幀的 4x4 位姿矩陣
│   │   ├── transforms_val.json
│   │   ├── transforms_test.json
│   │   └── train/, val/, test/         # 800x800 RGBA 圖片
│   ├── chair/
│   ├── drums/
│   └── ...
├── nerf_llff_data/                     # LLFF 前向場景資料集
│   ├── fern/
│   │   ├── poses_bounds.npy            # Nx17 矩陣 (3x5 位姿 + 2 bounds)
│   │   └── images/                     # 高解析度實拍照片
│   ├── flower/
│   └── ...
└── nerf_real_360/                      # Real 360 環繞資料集
    ├── pinecone/
    └── ...
```

### 2. 安裝依賴

```bash
pip install -r requirements.txt
```

需要的套件：`flask`、`numpy`、`opencv-python`

### 3. 啟動

```bash
python app.py
```

瀏覽器開啟 http://localhost:8080

## 使用方式

### 3D 瀏覽

1. 從左側下拉選單選擇資料集
2. **滾輪**縮放、**左鍵拖曳**旋轉、**右鍵拖曳**平移
3. **點擊**球體查看該相機的圖片和位姿資訊
4. 使用 checkbox 篩選 train / val / test

### 極線校驗

1. 載入資料集後，點擊左側的 **Verify Poses** 按鈕
2. 等待分析完成（LLFF ~12 秒 / Synthetic ~60 秒）
3. 球體自動變色，依極線誤差分級：
   - **綠色** (good)：< 2 px — 位姿精確
   - **黃色** (warning)：2–5 px — 輕微偏差
   - **紅色** (bad)：> 5 px — 顯著錯誤
   - **深灰** (skipped)：大型資料集抽樣時未檢測的相機
4. 點擊任一球體可在右側面板查看：極線誤差、評等、鄰近配對及各自誤差
5. 點擊 **Reset Colors** 恢復原始顏色

## 極線校驗演算法

校驗模組位於 `verification/epipolar.py`，獨立於主程式，可單獨引用。

### 原理

對於兩個相機 i 和 j，如果它們的位姿 (R, t) 和內參 K 都正確，那麼：
- 從位姿可以推算出 **Fundamental Matrix** F
- 圖片中的對應特徵點 (p₁, p₂) 應滿足極線約束：`p₂ᵀ · F · p₁ = 0`
- 即 p₂ 應精確落在極線 `l = F · p₁` 上

如果 p₂ 到極線的距離（單位：像素）很大，代表位姿可能有誤。

### 步驟

```
對每個相機 i：
  1. 找 3 個最近鄰相機 j（按歐氏距離）
  2. 對每個 pair (i, j)：
     a. 從已知位姿計算 F = K⁻ᵀ · [t]× · R · K⁻¹
     b. 讀取兩張圖片，SIFT 偵測特徵點
     c. Ratio test (0.7) + 雙向互相一致性過濾
     d. 對每對匹配計算對稱極線距離
     e. 取中位數作為該 pair 的誤差
  3. 該相機的最終分數 = 所有 pair 誤差的中位數
```

### 內參來源

| 格式 | 來源 | 計算方式 |
|------|------|----------|
| Synthetic | `transforms_train.json` 中的 `camera_angle_x` | `focal = 0.5 × width / tan(0.5 × camera_angle_x)` |
| LLFF | `poses_bounds.npy` 中 3×5 矩陣的第 5 列 | `[H, W, focal]`，按縮放比例調整 |

## 實踐過程與踩坑紀錄

### 坐標系轉換（最關鍵的坑）

極線幾何的計算需要在 **OpenCV 坐標系**（Y-down, Z-forward）下進行，但 NeRF 資料集使用不同的坐標系慣例：

**Synthetic NeRF** 使用 OpenGL 慣例：
- Y 軸朝上、-Z 軸朝前
- 轉換方式：將旋轉矩陣的 Y 和 Z 列取反

```python
M[:3, 1] *= -1  # negate Y column
M[:3, 2] *= -1  # negate Z column
```

**LLFF** 的 3×5 位姿矩陣列順序是 `[down, right, back]`，**不是** `[right, up, back]`：
- 這是開發過程中花最多時間除錯的地方
- 一開始假設列順序是 `[right, up, back]`，導致 LLFF 誤差高達 10–25px
- 修正後降至 ~0.2px

```python
# LLFF columns: [down, right, back] → OpenCV: [right, down, forward]
out[:3, 0] = R[:, 1]    # right
out[:3, 1] = R[:, 0]    # down
out[:3, 2] = -R[:, 2]   # forward = -back
```

### 特徵匹配的選擇

最初使用 **ORB + cross-check matching**，在合成資料集上表現極差：

| 方法 | LLFF fern 中位誤差 | Synthetic lego 中位誤差 | 結論 |
|------|-------------------|----------------------|------|
| ORB + cross-check | ~100 px | ~100 px | 完全不可用 |
| ORB + ratio test | 太少匹配 (6個) | — | 不穩定 |
| SIFT + ratio test + 雙向一致 | **~0.2 px** | **~0.3 px** | 準確可靠 |

問題出在合成圖片有大量重複紋理和鏡面反射，ORB 的二進位描述子區分不了相似區域，導致 ~80% 的匹配都是錯誤的。

切換到 SIFT 並加上 Lowe's ratio test（閾值 0.7）+ 雙向互相一致性檢查後，問題徹底解決。

### F 矩陣驗證方法

為了確認 F 矩陣計算正確，使用了「已知 3D 點投影」驗證法：

1. 在世界坐標中隨機生成 3D 點
2. 用兩個相機的投影矩陣分別投影到圖片坐標
3. 驗證投影點對滿足極線約束 `p₂ᵀ · F · p₁ ≈ 0`

兩種資料集的合成投影點誤差都是 **0.000000 px**，確認 F 矩陣推導正確，問題出在特徵匹配而非幾何計算。

### 大型資料集處理

Synthetic lego 有 400 張圖片（train 100 + val 100 + test 200），全部驗證需要處理 ~600 對圖片。每對需要 SIFT 偵測 + 匹配，約 0.5–1 秒，全部跑完要 ~10 分鐘。

解決方案：當相機數量超過 100 時，隨機抽樣 100 個相機進行驗證，未被抽到的標記為 `unverified`。

## 專案結構

```
.
├── app.py                    # Flask 伺服器、API 路由、資料集解析
├── static/index.html         # 單頁前端 (Three.js + vanilla JS)
├── verification/             # 獨立校驗模組（可單獨引用）
│   ├── __init__.py
│   └── epipolar.py           # 極線一致性校驗演算法
├── requirements.txt
├── dataset/                  # NeRF 資料集（不納入 git）
└── README.md
```

## API

| 端點 | 說明 | 回傳 |
|------|------|------|
| `GET /api/datasets` | 列出所有偵測到的資料集 | `[{name, path, type}]` |
| `GET /api/dataset/<path>` | 取得資料集的相機位姿 | `{cameras: [...], count}` |
| `GET /api/image/<path>` | 提供圖片檔案 | 圖片 |
| `GET /api/verify/<path>` | 執行極線校驗 | `{results: [{index, error, grade, pairs}]}` |

## 支援的資料集格式

| 格式 | 偵測方式 | 範例場景 |
|------|----------|----------|
| NeRF Synthetic | 存在 `transforms_train.json` | lego, chair, drums, hotdog, ficus, materials, mic, ship |
| LLFF | 存在 `poses_bounds.npy` | fern, flower, fortress, horns, leaves, orchids, room, trex |
| Real 360 | 存在 `poses_bounds.npy` | pinecone, vasedeck |

## 環境需求

- Python 3.8+
- Flask
- NumPy
- OpenCV (`opencv-python`)

## 參考

- [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://www.matthewtancik.com/nerf) (Mildenhall et al., ECCV 2020)
- [LLFF: Local Light Field Fusion](https://bmild.github.io/llff/) (Mildenhall et al., ACM TOG 2019)
