with open('./train2.txt','r',encoding = 'utf8') as f:
        for line in f.readlines():
            if line.__contains__('O'):
                print(line)