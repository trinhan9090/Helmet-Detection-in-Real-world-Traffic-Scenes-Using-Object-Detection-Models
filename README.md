# HDIRLT - Helmet Detection in Real-Life Traffic

Dự án tự học về phát hiện đối tượng trong bối cảnh giao thông thực tế, tập trung vào bài toán nhận diện mũ bảo hiểm và các đối tượng liên quan bằng mô hình YOLO.

Dự án hiện đang trong quá trình phát triển. Mục tiêu chính không phải là xây dựng một sản phẩm hoàn chỉnh ngay từ đầu, mà là thực hành quy trình làm việc với dữ liệu thị giác máy tính: khảo sát dataset, phân tích annotation, huấn luyện mô hình, đánh giá kết quả và rút ra nhận xét từ thực nghiệm.

## Mục tiêu dự án

- Tìm hiểu bài toán phát hiện mũ bảo hiểm trong môi trường giao thông thực tế.
- Làm quen với định dạng dữ liệu YOLO.
- Phân tích chất lượng dữ liệu và phân bố annotation.
- Huấn luyện mô hình phát hiện đối tượng bằng YOLO.
- Đánh giá hiệu năng mô hình qua các chỉ số như Precision, Recall, mAP50 và mAP50-95.
- Ghi lại quá trình học, thử nghiệm và các nhận xét phục vụ báo cáo/nghiên cứu cá nhân.

## Dataset

Dataset được lấy từ Roboflow Universe:

- Project: Helmet / Helmet9
- Format: YOLOv8
- Số ảnh: 2,121
- Kích thước ảnh: 1024 x 1024
- License: CC BY 4.0

Các tập dữ liệu đã được chia sẵn:

- `train`
- `valid`
- `test`

Các lớp trong dataset:

| ID | Class |
|---:|-------|
| 0 | bike |
| 1 | helmet |
| 2 | lisc |
| 3 | no_helmet |
| 4 | noise |
| 5 | rider |

Ghi chú: `lisc` được hiểu là biển số xe/license plate. Tên class hiện được giữ nguyên theo dataset gốc.

## Cấu trúc thư mục hiện tại

```text
HDIRLT/
├── datasource/
│   ├── train/
│   ├── valid/
│   ├── test/
│   ├── data.yaml
│   ├── README.dataset.txt
│   └── README.roboflow.txt
├── notebooks/
│   ├── 01_eda_dataset_overview.ipynb
│   ├── 02_annotation_analysis.ipynb
│   └── 03_training_experiments.ipynb
├── Report/
│   ├── img_src/
│   └── supplementery/
├── structure.txt
└── README.md
```

## Nội dung các notebook

### 01 - EDA Dataset Overview

Notebook này dùng để khảo sát tổng quan dataset:

- Đọc cấu hình `data.yaml`.
- Kiểm tra danh sách class.
- Kiểm tra kích thước ảnh.
- Thống kê số lượng ảnh, label và bounding box theo từng split.
- Phân tích phân bố class trong toàn bộ dataset và theo từng tập train/valid/test.
- Kiểm tra các file label rỗng.

Một số quan sát ban đầu:

- Dataset có 2,121 ảnh.
- Tất cả ảnh đã được resize về 1024 x 1024.
- Có 2 file label rỗng trong tập train.
- Các class có phân bố không hoàn toàn cân bằng; `rider` và `bike` xuất hiện nhiều hơn, trong khi `noise` ít hơn.

### 02 - Annotation Analysis

Notebook này tập trung vào phân tích annotation chi tiết hơn:

- Kiểm tra các ảnh có label rỗng.
- Trực quan hóa mẫu ảnh theo từng class.
- Phân tích diện tích bounding box.
- Phân tích tỷ lệ khung hình của bounding box.
- Phân tích vị trí tâm bounding box.
- Kiểm tra quan hệ không gian giữa các class, ví dụ:
  - `helmet -> rider`
  - `no_helmet -> rider`
  - `rider -> bike`
  - `lisc -> bike`

Class `noise` được xử lý như một nhãn cần khảo sát cẩn thận. Ở giai đoạn hiện tại, dự án chưa đưa ra kết luận tuyệt đối về bản chất của class này. Thay vào đó, notebook dùng phân tích hình học và quan hệ không gian để xem `noise` có khác biệt như thế nào so với các class chính.

### 03 - Training Experiments

Notebook này dùng để huấn luyện và đánh giá mô hình YOLO.

Mô hình đã thử nghiệm:

- YOLO11m
- Huấn luyện với 6 class
- Các thực nghiệm gồm baseline và cấu hình batch/epoch lớn hơn

Một số kết quả ghi nhận trong notebook:

- Trên validation set, thực nghiệm tốt nhất đạt khoảng:
  - mAP50: 0.934
  - mAP50-95: 0.715
- Trên test set:
  - mAP50 tổng thể: khoảng 0.883
  - mAP50-95 tổng thể: khoảng 0.680
  - Class `no_helmet` có kết quả thấp hơn một số class khác, nên cần được phân tích thêm.

## Hướng phát triển tiếp theo

Dự án vẫn đang tiếp tục hoàn thiện. Một số hướng có thể làm tiếp:

- Chuẩn hóa lại cấu trúc thư mục theo hướng dễ tái lập hơn.
- Tách code dùng lại từ notebook thành script trong `src/`.
- Bổ sung file cấu hình training trong `configs/`.
- Lưu kết quả train/evaluation vào `runs/`.
- Viết lại báo cáo chính thức từ các kết quả EDA và training.
- Phân tích sâu hơn lỗi dự đoán của class `helmet` và `no_helmet`.
- Làm rõ vai trò của class `noise` trong dataset.
- Thử nghiệm thêm các phiên bản YOLO khác hoặc chiến lược augmentation khác.

## Ghi chú

Dự án này thiên về học tập, thực hành và báo cáo nghiên cứu cá nhân. Vì vậy, notebook có thể vẫn còn các đoạn code thử nghiệm, đường dẫn Colab/Google Drive, hoặc phần phân tích chưa được đóng gói hoàn chỉnh.
