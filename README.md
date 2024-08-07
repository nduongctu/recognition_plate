# Nhận diện Biển số Xe

Repository này chứa ứng dụng web để nhận diện biển số xe theo thời gian thực bằng cách sử dụng camera hoặc tải lên video từ thiết bị. Ứng dụng cho phép người dùng chọn camera, kết nối và xử lý luồng video để phát hiện và nhận dạng biển số xe.

## Tính năng
- **Tải lên video**: Tải lên video để nhận dạng.
- **Chọn Camera**: Chọn giữa camera trước và camera sau để nhận dạng.
- **Xử lý Thời gian Thực**: Truyền và xử lý video theo thời gian thực để nhận dạng biển số xe.
- **Giao Diện Người Dùng**: Giao diện trực quan cho việc chọn camera và truyền video.

## Cài đặt

1. Clone repository:
   git clone https://github.com/nduongctu/recognition_plate.git
2. Di chuyển vào thư mục dự án:
  cd recognition_plate
3. Cài đặt các thư viện cần thiết:
   pip install -r requirements.txt
4. Cài đặt cuda
   - Tải xuống CUDA Toolkit: Truy cập trang web NVIDIA Developer để tải xuống CUDA Toolkit 12.1 phù hợp với hệ điều hành của bạn.
   - Cài đặt PyTorch:
      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
5. Cập nhật địa chỉ IP ở file app.py và index_draw.html
6. chạy python app.py
7. chạy file index_draw.html hoặc truy cập bằng địa chỉ IP
