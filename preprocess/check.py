import os
from glob import glob
now_dir = os.getcwd()
up_dir, _ = os.path.split(now_dir)
dic = {}
all_txt = []
with open(up_dir+'/librilabel', 'r') as f:
    lines = f.readlines()#.split('\n')
    
    for line in lines:
        #txt = line.strip().split('\t')#rstrip('\n')
        txt = line.strip()
        all_txt.append(txt)
        #print('line {} txt {}'.format(line, txt))
    for i in range(len(all_txt)):
        #print("{}th txt {}".format(i, all_txt[i]))
        this = all_txt[i].split("\t")
        #print("this", this)
        #print(len(this))
        num = this[0]
        try:
            char = this[1]
        except:
            char = ' '
        
        print("{}th num {} char {}".format(i, num, char))
        dic[char] = i
print(dic)

all_train_txt = glob('/sdd_temp/junewoo10/LibriSpeech/train-clean-360/*/*/*.txt')
for i in range(len(all_train_txt)):
    with open(all_train_txt[i], 'r') as f:
        one_line = f.readline()
        one_line = one_line[17:-1]
        print(one_line)
        txt = ''
        for i in range(len(one_line)):
            char = dic[one_line[i]]
            print(char)
            txt += str(char)+' '
        print('txt is', txt)




        exit()
