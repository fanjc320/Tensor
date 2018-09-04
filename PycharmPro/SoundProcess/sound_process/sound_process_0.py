import os
 
filepath = "./data/" #添加路径
filename= os.listdir(filepath) #得到文件夹下的所有文件名称 
for file in filename:
    print(filepath+file)
	
# 　这里用到字符串路径：

    # 1.通常意义字符串(str)
    # 2.原始字符串，以大写R 或 小写r开始，r''，不对特殊字符进行转义
    # 3.Unicode字符串，u'' basestring子类
	
# path = './file/n'
# path = r'.\file\n'
# path = '.\\file\\n'
# 三者等价，右划线\为转义字符，引号前加r表示原始字符串，而不转义（r:raw string）.

