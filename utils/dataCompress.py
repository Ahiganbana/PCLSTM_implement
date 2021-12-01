import pandas as pd
import os

step = 2000
filepath = r"D:\Learning\data\211128\backup\awacs"
newfilepath = r"D:\Learning\data\211128\compress\awacs"
if not os.path.exists(newfilepath):
    os.makedirs(newfilepath)
for roots, dirs, files in os.walk(filepath):
    for name in files:
        fpath = os.path.join(filepath, name)
        csv = pd.read_csv(fpath, encoding="gbk", header=0)
        #csv_reader = csv.reader(open(fpath))
        l = len(csv)
        s = int(l / step)
        csv = csv.values.tolist()
        print(type(csv))
        print(len(csv))
        if(l < step):
            continue;
        csv_writer = []
        for i in range(0, l, s):
            csv_writer.append(csv[i])
        if(len(csv_writer) > step) :
            csv_writer = csv_writer[:step]
        newpath = os.path.join(newfilepath, 'p'+name)
        print(newpath)
        df = pd.DataFrame(csv_writer)
        df.to_csv(newpath,header=None, index=None)
        # with open(newpath,"w", newline='') as csvfile:
        #     writer = csv.writer(csvfile)
        #     writer.writerow(C)
        #     writer.writerows(csv_writer)