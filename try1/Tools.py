import json
import os

def readFile_JSON(link, filename, encoding):
    # 假设内存足够大，不考虑内存泄漏
    # link 文件路径
    # filename 无后缀文件名
    # 直接抛出io异常，不作任何处理

    # 读取出异常，多半是路径格式不对，注意转义
    # 强制文件后缀过滤，注意文件保存格式，我有洁癖
    src = link + "/" + filename + ".json"
    # print(src)
    file = open(src, "r", encoding=encoding)
    fileData = file.read()
    return json.loads(fileData)


def foreachJSON(JSON, func):
    data = JSON["data"]

    res = []
    for i in range(0, len(data)):
        # 特征提取工作
        # 可自主选择func(dataToFunc)函数传入，多态，还没掌握py的面向对象语法，凑合一下
        res.append(func(data[i]))
    return res


def foreachFolder(link, encoding, func):
    # 遍历某一文件夹的子文件
    # dict
    filesName = getFilesName(link)
    jsonList = []
    for i in range(0, len(filesName)):
        temp = readFile_JSON(link, filesName[i], encoding)
        temp = foreachJSON(temp, func)
        for k in range(0, len(temp)):
            jsonList.append(temp[k])
    return jsonList


def getFilesName(link):
    filesName = os.listdir(link)
    nameList = []
    for i in range(0, len(filesName)):
        # 文件夹过滤
        if (os.path.isfile(link + "/" + filesName[i])):
            temp = filesName[i].split(".")
            # 只获取[*.json]的文件名,过滤器
            if (len(temp) == 2 and temp[1] == "json"):
                nameList.append(temp[0])
    return nameList

def writeJSONList(jsonList, targetLink, targetFilename, encoding):
    src = targetLink + "/" + targetFilename + ".json"
    writeFile(src, json.dumps(jsonList), encoding)

def writeCsv(content, targetLink, targetFilename, encoding):
    src = targetLink + "/" + targetFilename + ".csv"
    writeFile(src, content, encoding)

def writeFile(src, content, encoding):
    file = open(src, "w", encoding=encoding)
    file.write(content)
    file.close()
