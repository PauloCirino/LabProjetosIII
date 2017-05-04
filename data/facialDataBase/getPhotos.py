# ----------------------------------------------------------------
#
#
# This program access a website list of top famous people
# and download to a local folder the first 100
# photos of the google search of each name of the list to
# create a database for training the facial classification
# software.
#

# -----
# Import the necessary URL and RE functions
from urllib import urlopen
from re import findall
from bs4 import BeautifulSoup
import requests
import re
import urllib2
import os
import cookielib
import json

# -----
# Define the list URL
url = 'http://people.com/celebrities/'


# -----
# Get a link to the web page from the server, using
# the URL above
List_page = urlopen(url)

# -----
# Extract the web page's content as a string
html_code = List_page.read()

# ----
# Close the connection to the web server
List_page.close()

# -----
# Find the names in the list.
# We assume the names are located in the following html document place:
#
#    ">[a-zA-Z]+.[a-zA-Z]+</a></dd>
#
#
names_list = findall('">[a-zA-Z]+.[a-zA-Z]+</a></dd>', html_code)

# ----
# Discard the characters taken from the html code that are not part
# of people names:
s = len(names_list)
names_list2 = range(s)

for x in range(s):
    names_list2[x] = names_list[x][2:-9]

print names_list2

# ----
# Get the images from a Google search:
def get_soup(url,header):
    return BeautifulSoup(urllib2.urlopen(urllib2.Request(url,headers=header)),'html.parser')

for y in range(99,s):
    query = names_list2[y]
    image_type="ActiOn"
    query= query.split()
    query='+'.join(query)
    url="https://www.google.co.in/search?q="+query+"&source=lnms&tbm=isch"
    print url
    #add the directory for your image here
    DIR="Pictures"
    header={'User-Agent':"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"
    }
    soup = get_soup(url,header)


    ActualImages=[]# contains the link for Large original images, type of  image
    for a in soup.find_all("div",{"class":"rg_meta"}):
        link , Type =json.loads(a.text)["ou"]  ,json.loads(a.text)["ity"]
        ActualImages.append((link,Type))

    print  "there are total" , len(ActualImages),"images"

    if not os.path.exists(DIR):
                os.mkdir(DIR)
    DIR = os.path.join(DIR, query.split()[0])

    if not os.path.exists(DIR):
                os.mkdir(DIR)
    ###print images
    for i , (img , Type) in enumerate( ActualImages):
        try:
            req = urllib2.Request(img, headers={'User-Agent' : header})
            raw_img = urllib2.urlopen(req).read()

            cntr = len([i for i in os.listdir(DIR) if image_type in i]) + 1
            print cntr
            if len(Type)==0:
                f = open(os.path.join(DIR , image_type + "_"+ str(cntr)+".jpg"), 'wb')
            else :
                f = open(os.path.join(DIR , image_type + "_"+ str(cntr)+"."+Type), 'wb')


            f.write(raw_img)
            f.close()
        except Exception as e:
            print "could not load : "+img
            print e