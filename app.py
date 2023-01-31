#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：table-text-recognition 
@File    ：app.py.py
@Author  ：AkiiLucky
@Date    ：2023/1/31 18:00 
'''


from flask import Flask, render_template, request, redirect, url_for
from flask import make_response, json, jsonify
from ocr.table_ocr import table_ocr, itemFilter

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False


@app.route('/')
def fun():
    return render_template('test.html')

@app.route('/table/ocr', methods=['GET', 'POST'])
def tableOcr():
    # 通过表单中name值获取图片
    img = request.files['file']
    # 保存图片
    path = f'./static/images/image.jpg'
    img.save(path)
    res = table_ocr(path, baiduOcr=False, DEBUG_MODE=False)
    res = itemFilter(res)
    return jsonify(res)


if __name__ == '__main__':
    app.run()