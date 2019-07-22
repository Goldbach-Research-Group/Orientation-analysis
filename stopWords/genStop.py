import help
import json

str=help.readTXT("stopWordList(sou).txt")
list=str.split("\n")
jsonlist={}
nowhead=""
for word in list:
    anowhead=help.gethead(word)
    if nowhead!=anowhead:
        nowhead=anowhead
    if not nowhead in jsonlist.keys():
        jsonlist[nowhead]=[]
    jsonlist[nowhead].append(word)

jsoncode=json.dumps(jsonlist,ensure_ascii=False)
help.writeTXT("stopWordList(gen).txt",jsoncode)