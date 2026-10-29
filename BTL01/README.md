# BTL01 — Xấp xỉ hàm mật độ Gaussian 2D bằng MLP (PyTorch)

Bài tập lớn môn Học sâu: xấp xỉ hàm mật độ Gaussian hai chiều bằng mạng perceptron đa tầng (MLP) cài đặt thuần PyTorch, kèm ba nhóm thực nghiệm có kiểm soát về quy mô dữ liệu (A), độ rộng tầng ẩn (B) và độ sâu (C).

## Cấu trúc thư mục

```
BTL01/
├── BTL01_MLP_Gaussian.ipynb   # Notebook chính, chia mục 0–9
├── requirements.txt
├── README.md
├── report/                    # Báo cáo Word/PDF + nguồn LaTeX
│   ├── BTL01_Report.tex       # Nguồn LaTeX của báo cáo
│   ├── BTL01_Report.docx      # Báo cáo Word (19 trang)
│   ├── BTL01_Report_Final.pdf # Bản PDF xuất từ Word
│   ├── make_report_figures.py # Sinh lại 14 hình cho báo cáo
│   ├── reference.docx         # Template style cho Pandoc
│   └── figures/               # Hình dùng trong báo cáo
└── outputs/
    ├── data/                  # train_pool.csv, validation.csv, test.csv
    ├── figures/               # 01–14*.png (hình của notebook)
    ├── logs/                  # config, results_*.csv, history_*.csv, summary_all.csv
    └── models/
        ├── runs/              # checkpoint từng lượt chạy + baseline.pt
        └── best_model.pt      # mô hình cuối được chọn
```

## Môi trường

- Python 3.10–3.12 (khuyến nghị), PyTorch CPU (đã kiểm chứng với `torch 2.14.0+cpu`, `numpy 2.5.3`, `pandas 3.0.5`, `matplotlib 3.11.2`).
- Cài đặt:

```bash
pip install -r requirements.txt
```

## Cách chạy

Mở notebook và chạy **Run All** từ thư mục `BTL01/`:

```bash
jupyter notebook BTL01_MLP_Gaussian.ipynb
```

Hoặc chạy headless (khoảng 10–15 phút trên CPU):

```bash
jupyter nbconvert --to notebook --execute --inplace BTL01_MLP_Gaussian.ipynb
```

Notebook dùng đường dẫn tương đối theo thư mục hiện hành, vì vậy cần chạy trong `BTL01/`. Kernel cần có PyTorch; notebook đã khai báo kernel `deep_learning` nhưng có thể đổi sang kernel Python bất kỳ có đủ thư viện.

## Báo cáo Word/PDF

Báo cáo được viết bằng LaTeX (`report/BTL01_Report.tex`) rồi biên dịch sang Word bằng Pandoc, sau đó dùng Microsoft Word để cập nhật mục lục và xuất PDF. Để tái tạo:

```bash
# 1. Sinh lại 14 hình cho báo cáo (đọc từ outputs/)
python report/make_report_figures.py

# 2. LaTeX -> DOCX (chạy trong thư mục BTL01/)
pandoc report/BTL01_Report.tex -o report/BTL01_Report.docx \
  --reference-doc=report/reference.docx --toc --toc-depth=2

# 3. Mở BTL01_Report.docx bằng Word, Update Table of Contents,
#    Save và Export as PDF (BTL01_Report_Final.pdf)
```

Bản đã biên dịch sẵn: `report/BTL01_Report.docx` (19 trang) và `report/BTL01_Report_Final.pdf`. Lưu ý tên `BTL01_Report.pdf` (nếu còn) là bản cũ; dùng `BTL01_Report_Final.pdf`.

## Bài toán

- Hàm mật độ Gaussian 2D với $\mu=(0,0)$, $C=\begin{bmatrix}0.3 & -0.5\\-0.5 & 2\end{bmatrix}$.
- Miền lấy mẫu: $x_1\in[-2,2]$, $x_2\in[-4,4]$; nhãn $y=f(x_1,x_2)$ không nhiễu.
- Master train pool 2000 mẫu, validation 1000 mẫu, test 100 mẫu (độc lập, seed cố định).
- MLP 2 đầu vào – $C$ tầng ẩn × $B$ nơ-ron – 1 đầu ra, mất mát MSE.

## Cấu hình baseline

| Tham số | Giá trị |
|---|---|
| Số mẫu train | 1000 |
| Kiến trúc | 2 tầng ẩn × 64 nơ-ron |
| Kích hoạt | `tanh` |
| Optimizer | Adam, lr $=5\times10^{-3}$ |
| Batch size | 64 |
| Epoch tối đa | 300 |
| Early stopping | patience 50 theo validation loss |
| Test | 100 mẫu, chỉ đánh giá cuối |

## Ba nhóm thực nghiệm

| Nhóm | Biến độc lập | Giá trị | Cố định |
|---|---|---|---|
| A | Số mẫu train $A$ | 200, 400, …, 2000 | 2×64, các siêu tham số baseline |
| B | Số nơ-ron 1 tầng ẩn $B$ | 1, 2, …, 10 | $A=1000$, $C=1$ |
| C | Số tầng ẩn $C$ | 1, 2, 3, 4 | $A=1000$, $B=64$ |

Mỗi cấu hình chạy 3 seed (42, 43, 44), báo cáo trung bình ± độ lệch chuẩn. Tập con train của thực nghiệm A được lấy lồng nhau từ cùng master pool để so sánh công bằng. Test không tham gia chọn siêu tham số.

## Kết quả chính

| Cấu hình | Test MSE | Test MAE |
|---|---|---|
| Baseline (A=1000, 2×64) | $3.26\times10^{-6}$ | $1.37\times10^{-3}$ |
| Mô hình cuối (2×64, 2000 mẫu) | $1.09\times10^{-6}$ | $7.97\times10^{-4}$ |

- A: test MSE giảm khoảng 19 lần từ $A=200$ đến $2000$, bão hòa quanh $A\approx1000$–$1200$.
- B: $B\le3$ thiếu khớp mạnh; sai số giảm đơn điệu đến $B=10$ nhưng chưa bão hòa; chưa thấy quá khớp.
- C: $C=1$–$2$ đã đủ tốt; $C=3$–$4$ chỉ cải thiện nhẹ.

## Đầu ra

- `outputs/logs/results_A.csv`, `results_B.csv`, `results_C.csv`: từng lượt chạy (config, seed, metrics, thời gian).
- `outputs/logs/history_*.csv`: loss theo epoch.
- `outputs/logs/summary_all.csv`: bảng tổng hợp để chọn mô hình theo validation.
- `outputs/logs/results_baseline.csv`, `final_test_results.csv`: kết quả baseline và đánh giá test cuối.
- `outputs/figures/`: 14 hình (dữ liệu, loss, y–ŷ, bề mặt, so sánh A/B/C, phần dư).
- `outputs/models/runs/*.pt`: checkpoint tốt nhất từng lượt chạy; `outputs/models/best_model.pt`: mô hình cuối.

## Tái lập

Mọi seed được khai báo trong notebook (`SEED = 42`, seed dữ liệu 43/44/45, seed huấn luyện 42/43/44). Chạy lại từ đầu sẽ tái tạo cùng dữ liệu, cùng kết quả trên CPU (sai khác nhỏ có thể do thứ tự tính toán, nhưng đã cố định seed và generator của DataLoader).
