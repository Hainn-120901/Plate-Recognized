import os

import cv2
import numpy as np
from random import choice
import socket
import requests
from PIL import Image
from flask import Flask, jsonify, request
from io import BytesIO
from flask_cors import CORS, cross_origin
from flask_httpauth import HTTPBasicAuth
import base64
from config import *
import io
from functools import lru_cache
from license_plates import LicensePlate

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
model = LicensePlate()


def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        ip_sv = s.getsockname()[0]
    except:
        ip_sv = '127.0.0.1'
    finally:
        s.close()
    return ip_sv


desktop_agents = [
    'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_1) AppleWebKit/602.2.14 (KHTML, like Gecko) Version/10.0.1 Safari/602.2.14',
    'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.71 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.98 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.98 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.71 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; WOW64; rv:50.0) Gecko/20100101 Firefox/50.0']


def random_headers():
    return {'User-Agent': choice(desktop_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'}


def download_image(image_url):
    header = random_headers()
    response = requests.get(image_url, headers=header, stream=True, verify=False, timeout=5)

    image = Image.open(BytesIO(response.content)).convert('RGB')

    return image


def jsonify_str(output_list):
    with app.app_context():
        with app.test_request_context():
            result = jsonify(output_list)
    return result


app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
auth = HTTPBasicAuth()


def create_response_v2(data, error_code, error_message):
    if data == "":
        response = {
            "errorCode": error_code,
            "errorMessage": error_message,
            "data": data
        }
    else:
        response = {
            "errorCode": error_code,
            "errorMessage": error_message,
            "data": data
        }
    return response

def get_byte_img(img_new):
    img_color = cv2.cvtColor(img_new, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_color)
    img_byte_arr = io.BytesIO()
    img_pil.save(img_byte_arr, format='JPEG')
    encoded_img = base64.encodebytes(img_byte_arr.getvalue()).decode('ascii')
    return encoded_img


@lru_cache(maxsize=1)
@app.route("/license_plate", methods=['GET', 'POST'])
@cross_origin()
def card():
    format_type = request.args.get('format_type', default="file", type=str)

    if request.method == "POST":
        if format_type not in params_post:
            return jsonify_str(create_response_v2("", "6", "Incorrect format type"))
        if format_type == "file":
            try:
                data_img = request.files["img"].read()
                img = Image.open(BytesIO(data_img)).convert('RGB')
            except:
                return jsonify_str(create_response_v2("", "3", "Incorrect image format"))
        else:
            try:
                content = request.get_json()
                image_encode = content['img']
                img = base64.b64decode(str(image_encode))
                img = Image.open(io.BytesIO(img)).convert('RGB')
            except:
                return jsonify_str(create_response_v2("", "3", "Incorrect image format"))
    else:
        if format_type not in params_get:
            return jsonify_str(create_response_v2("", "6", "Incorrect format type"))
        try:
            image_url = request.args.get('img', default='', type=str)
            img = download_image(image_url)
        except:
            return jsonify_str(create_response_v2("", "2", "Url is unavailable"))

    image_input = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    list_result, list_image = model.predict_plate(image_input)
    if len(list_result) > 0:
        list_image_base64 = [get_byte_img(imgbase64) for imgbase64 in list_image]
        result_json = {
            "result": list_result,
            "image": list_image_base64
        }
    else:
        result_json = None

    if result_json is None:
        return jsonify_str(create_response_v2("", "1", "The photo does not contain content"))

    return jsonify_str(create_response_v2(result_json, "0", "Success"))


app.run("0.0.0.0", 1904, threaded=False, debug=False)