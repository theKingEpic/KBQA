# with open('./train2.txt','r',encoding = 'utf8') as f:
#         for line in f.readlines():
#             if line.__contains__('O'):
#                 print(line)


#此文件是为了处理数据集
#step1
# with open("./test.txt","r",encoding="utf-8") as f:
#     lines = f.readlines()
# with open("./test.txt","w",encoding="utf-8") as f_w:
#     for line in lines:
#         if "O" in line:
#             continue
#         f_w.write(line)

# #step2
# with open("./test.txt","r",encoding="utf-8") as f:
#     lines = f.readlines()
# with open("./test.txt","w",encoding="utf-8") as f_w:
#     for line in lines:
#         if line=='\n':
#             continue
#         else:
#             f_w.write(line)

# #step3        
# with open("./test.txt","r",encoding="utf-8") as f:
#     lines = f.readlines()
# with open("./test.txt","w",encoding="utf-8") as f_w:
#     for line in lines:
#         if "B" in line:
#              f_w.write("\n"+line)
#         else:
#             f_w.write(line)