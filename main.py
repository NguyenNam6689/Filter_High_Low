import matplotlib.pyplot as plt
import cv2

img = cv2.imread('./images/beach.png')
print(f"Chiều rộng của hình ảnh: {img.shape[1]} pixel")
print(f"Chiều cao của hình ảnh: {img.shape[0]} pixel")
print(f"Số kênh của hình ảnh: {img.shape[2]}")

fig = plt.figure()
plt.imshow(img)
plt.show()

cv2.imwrite("test_image.png", img)
