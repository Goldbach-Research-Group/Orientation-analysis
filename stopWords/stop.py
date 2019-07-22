import json
import stopWords.help as help

def init(path):
    jsoncode=help.readTXT(path)
    global stoplist
    stoplist=json.loads(jsoncode)

def isStopWord(word):
    nowhead=help.gethead(word)
    if nowhead in stoplist.keys():
        if word in stoplist[nowhead]:
            return True
    return False
