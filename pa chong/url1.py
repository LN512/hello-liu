from urllib.request import urlopen
import re
import csv
class Spider:
    def __init__(self):
        self.URL = "https://www.google.com"
    def getLinks(self):
        url = self.URL
        website = urlopen(url)
        data = website.read().decode('gbk')
        links = re.findall('(?<=href=")[^"]*',data)
        saveLinks(links)
def saveLinks(links):
    f = open("links.csv",'w',newline='\n')
    foo = csv.writer(f)
    for link in links:
        if(link[0:4]=='http'):
            foo.writerow([link])
    f.close()

if __name__=='__main__':
    spider = Spider()
    spider.getLinks()