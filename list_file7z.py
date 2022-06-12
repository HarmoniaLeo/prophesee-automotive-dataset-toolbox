import pylzma
import py7zlib

for i in range(0,15):
    fp = open("/data/lbd/Large_Automotive_Detection_Dataset/train/trainfilelist" +  str(i).zfill(2) + ".7z",'rb')
    #生成一个archive对象
    archive = py7zlib.Archive7z(fp)

    #读取文件中所有的文件名
    names = archive.getnames()

    files_skip = [
        "moorea_2019-06-17_test_02_000_1037500000_1097500000",
        "moorea_2019-06-14_002_1220500000_1280500000",
        "moorea_2019-06-21_000_61500000_121500000",
        "moorea_2019-06-17_test_01_000_732500000_792500000",
        "moorea_2019-06-21_001_2196500000_2256500000",
        "moorea_2019-06-21_000_244500000_304500000",
        "moorea_2019-06-17_test_02_000_244500000_304500000",
        "moorea_2019-06-21_000_427500000_487500000",
        "moorea_2019-06-17_test_02_000_1830500000_1890500000",
        "moorea_2019-06-17_test_02_000_183500000_243500000",
        "moorea_2019-06-17_test_01_000_2135500000_2195500000",
        "moorea_2019-06-19_000_2562500000_2622500000",
        "moorea_2019-06-17_test_02_000_2623500000_2683500000",
        "moorea_2019-06-17_test_02_000_2013500000_2073500000",
        "moorea_2019-06-17_test_02_000_1769500000_1829500000"
    ]

    for file in files_skip:
        for name in names:
            if file in name:
                print(file, i)