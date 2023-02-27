from PIL import Image
import os
if __name__ == '__main__':
    train_dataset_path = "/home/b311/data/wdh/datasets/Samiler/train/"
    count = 0
    num = 0
    for filename1 in os.listdir(train_dataset_path):
        if filename1 != "list.txt":
            file_path = train_dataset_path + filename1 + "/00000001.jpg"

            img = Image.open(file_path)

            imgSize = img.size  #大小/尺寸
            if imgSize ==(1920,1080):
                print(filename1)
                num += 1
            count += 1
    print("共：",num)
