#!/usr/bin/env pytho
# -*- coding: utf-8 -*-

import requests
import os


def get_pages(keyword, pages, headers):
    params = []
    for i in range(30, 30*pages+30, 30):
        # 通过网上资料，可以使用 requests.get() 解析 json 数据，能够得到对应 url
        # 其中一个坑是，原来并不是以下的每个数据都是需要的，某些不要也可以！
        params.append({
                      'tn': 'resultjson_com',
                      'ipn': 'rj',
                      'ct': 201326592,
                      'is': '',
                      'fp': 'result',
                      'queryWord': keyword,
                      'cl': 2,
                      'lm': -1,
                      'ie': 'utf-8',
                      'oe': 'utf-8',
                      'adpicid': '',
                      'st': -1,
                      'z': '',
                      'ic': 0,
                      'word': keyword,
                      's': '',
                      'se': '',
                      'tab': '',
                      'width': '',
                      'height': '',
                      'face': 0,
                      'istype': 2,
                      'qc': '',
                      'nc': 1,
                      'fr': '',
                      'pn': i,
                      'rn': 30,
                      #'gsm': '1e',
                      #'1488942260214': ''
                  })
    url = 'https://image.baidu.com/search/acjson'
    urls = []
    for param in params:
        # url 与 param 合成完整 url
        try:
            urls.append(requests.get(url, param, headers=headers, timeout=3).json().get('data'))
        except ValueError:
            pass
    return urls


def get_img(data_list, local_path):

    if not os.path.exists(local_path):  # 新建文件夹
        os.makedirs(local_path)

    x = 0
    for data in data_list:
        for i in data:
            if i.get('thumbURL') != None:
                print('正在下载：%s' % i.get('thumbURL').encode('utf-8'))
                ir = requests.get(i.get('thumbURL'))
                open(local_path + '%04d.jpg' % x, 'wb').write(ir.content)
                x += 1
            else:
                print('图片链接不存在...')


if __name__ == '__main__':
    # keywords = ['卧室', '卫生间']
    # keywords = ['客厅', '厨房']
    keywords = ['厨房']
    kw_dict = {
               '卧室': 'bedroom',
               '卫生间': 'bathroom',
               '客厅': 'livingroom',
               '厨房': 'kitchen',
              }
    pages = 100
    headers = {'user-agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36'}
    for keyword in keywords:
        local_path = './downloads/%s/' % kw_dict[keyword]

        data_list = get_pages(keyword, pages, headers)
        get_img(data_list, local_path)

