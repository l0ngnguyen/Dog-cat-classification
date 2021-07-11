import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser(description='get dataframe store info about image from directory')
parser.add_argument('path', type=str, help='training directory store all image dog and cat', nargs=1)
parser.add_argument('--o', required=False, nargs=1)
args = parser.parse_args()

def readfile():
    """Nhận đầu vào là một path, đọc hết thông tin ảnh, sau đó
       trích xuất số lượng ảnh trong thư mục này bằng size 
       sao cho số lượng chó và mèo bằng nhau

    Args:
        train_path (string): đường dẫn đến thư mục chứa ảnh
        size (Int): số lượng ảnh muốn lấy ra

    Returns:
        Dataframe: chứa dữ liệu về tên ảnh và lớp của chúng 
    """
    # return list file name trong train_path
    filenames = os.listdir(args.path[0])
    categories = []
    for file in filenames:
        if file.split('.')[0] == 'dog':
            categories.append('1')
        else:
            categories.append('0')
    df = pd.DataFrame({'file name': filenames,
                       'category': categories})

    df = df.sample(frac=1).reset_index(drop=True)

    if args.o is not None:
        df.to_csv(args.o[0], index=False)
    else:
        df.to_csv('dog_cat_info.csv', index=False)

    return df

readfile()