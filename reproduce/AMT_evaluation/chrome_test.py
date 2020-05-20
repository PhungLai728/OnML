import os
from selenium import webdriver
# from PIL import Image
from selenium.webdriver.chrome.options import Options
import time
from optparse import OptionParser

############## Error note: 
'''

Message: session not created: This version of ChromeDriver only supports Chrome version 74
  (Driver info: chromedriver=74.0.3729.6 (255758eccf3d244491b8a1317aa76e1ce10d57e9-refs/branch-heads/3729@{#29}),platform=Mac OS X 10.13.6 x86_64)
When u see this message, download the chromdriver that is compatible with ur current Chrome version
'''
files = os.listdir('./Vis/')
# CHROME_PATH = '/usr/bin/google-chrome'
CHROMEDRIVER_PATH = '/Users/phunglai/Documents/Work/Papers/IJCNN2020/IJCNN2020_code/reproduce/AMT_evaluation/chromedriver-3'
WINDOW_SIZE = "1920,1080"
# WINDOW_SIZE = "1500,600"

count = 0
for f in files:
    count +=1
    print(count)
# for i in range(2):
    # f = files[i]
    print(f)
    url = 'file:///Users/phunglai/Documents/Work/Papers/IJCNN2020/IJCNN2020_code/reproduce/AMT_evaluation/Vis/' + f
    # print(url)
    img_name = f.split('.')[0] + '.png'
    # img_name = f.split('_')[0] + '_' + f.split('_')[1] + f.split('_')[2] + '.png'
    # print(img_name)
    #Open another headless browser with height extracted above
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--window-size=%s" % WINDOW_SIZE)
    chrome_options.add_argument("--hide-scrollbars")
    # chrome_options.binary_location = CHROME_PATH
    driver = webdriver.Chrome(
        executable_path=CHROMEDRIVER_PATH,
        options=chrome_options
    )
    # driver = webdriver.Chrome(executable_path='/Users/phunglai/Documents/Work/LIME/lime_original/lime/lime_ontology/OSIL_drug/Result/withOLLIE/3algs_rules/chromedriver', options=chrome_options)

    driver.get(url)
    #pause 3 second to let page loads
    time.sleep(0.5)
    #save screenshot
    driver.save_screenshot('./screen_shots/'+img_name)
    driver.close()
    print('ok')


# # driver.quit()
# print('ok')




# chromedriver = "/Users/phunglai/Documents/Work/LIME/lime_original/lime/lime_ontology/OSIL_drug/Result/withOLLIE/3algs_rules/chromedriver"
# os.environ["webdriver.chrome.driver"] = chromedriver
# driver = webdriver.Chrome(chromedriver)
# driver.get(url)
# time.sleep(0.5)
# #save screenshot
# driver.save_screenshot('./screen_shots/'+img_name)
# driver.close()
# print('ok')