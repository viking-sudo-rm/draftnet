import requests
import os
import bs4

os.makedirs('hero_icons', exist_ok=True)

url = 'http://www.dota2.com/heroes/'

res = requests.get(url) # download the page with the given url.
res.raise_for_status()

soup = bs4.BeautifulSoup(res.text, "html.parser")

a = soup.select('a')

#print(a[50].select('img')[0].get('src'))

# first hero is earthshaker, a[37]


for x in range(37, 150):
    # get the name of the hero
    name = a[x].get('id').replace('link_', '') + '.png'
    # create the file where we are going to save the image.
    imageFile = open(os.path.join('hero_icons', os.path.basename(name)), 'wb')
    # get the url for the image
    url = a[x].select('img')[0].get('src')
    # download the page
    res = requests.get(url, "html.parser")
    res.raise_for_status()
    # save image in our file
    for chunk in res.iter_content(1000000):
        imageFile.write(chunk)
    # close the file
    imageFile.close()

print('done')
