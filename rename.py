import os
import shutil

# Đường dẫn tới folder chứa các file ảnh của bạn
folder_path = '/home/saplab/Documents/paper_stream/test_data'

# Lấy danh sách các file trong folder
files = os.listdir(folder_path)

# Khởi đầu index từ 1000
index = 0

# Duyệt qua từng file và đổi tên
for file_name in files:
    # Kiểm tra nếu là file ảnh (có thể kiểm tra đuôi file)
    if file_name.endswith('.png') or file_name.endswith('.jpg') or file_name.endswith('.jpeg'):
        # Đặt tên mới cho file
        new_file_name = f'{index}.png'
        
        # Đường dẫn đến file gốc và file mới
        old_file_path = os.path.join(folder_path, file_name)
        new_file_path = os.path.join(folder_path, new_file_name)
        
        # Đổi tên file
        os.rename(old_file_path, new_file_path)
        
        # Tăng index lên cho file tiếp theo
        index += 1

print("Đổi tên các file thành công.")
