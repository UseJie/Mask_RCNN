import os

def batch_rename(path):
    # 批处理改名#
    # 原文件名：cat.0.jpg
    # 改为文件名：cat_0.jpg
     filenames = os.listdir(path)
     for filename in filenames:
         filename_list = filename.split('.')
         if filename_list[0] != 'cat' or filename_list[2] == 'json':
             continue
         new_filename = filename_list[0]+'_'+filename_list[1]+'.jpg'
         os.rename(os.path.join(filename), os.path.join(new_filename))
         print(new_filename)

if __name__ == '__main__':
    batch_rename('.')
