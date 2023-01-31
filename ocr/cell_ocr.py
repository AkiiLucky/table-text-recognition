import requests
import json
import base64


API_KEY = "pMEIQuzqnsLHLkdkIA1e19GK"
SECRET_KEY = "t7VKeb8XU7Hahpe68wRi5ybCbmHhhUkF"

def cell_ocr(img):
        
    url = "https://aip.baidubce.com/rest/2.0/ocr/v1/handwriting?access_token=" + get_access_token()

    data = {
        'image': 'data:image/jpeg;base64,'+str(img)[2:-1],
        'image_url': '',
        'type': 'https://aip.baidubce.com/rest/2.0/ocr/v1/general_basic',
        'detect_direction': 'false'
    }

    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': 'application/json'
    }
    
    response = requests.request("POST", url, headers=headers, data=data)
    # print(response.text)

    result_list = response.json()['words_result']
    if result_list:
        text = ""
        for r in result_list:
            text += r['words'].strip()
            text += " " #空格作为分割符
        return text
    else:
        return ""

    
    

def get_access_token():
    """
    使用 AK，SK 生成鉴权签名（Access Token）
    :return: access_token，或是None(如果错误)
    """
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": API_KEY, "client_secret": SECRET_KEY}
    return str(requests.post(url, params=params).json().get("access_token"))


if __name__ == '__main__':
    img_path = "img/cell9.png"
    img = None
    with open(img_path, 'rb') as f:
        img = base64.b64encode(f.read())


    print(cell_ocr(img))
