#!/usr/bin/python
# -*- coding: UTF-8 -*-

from flask import request, Flask, Response
from image_caption.build_vocab import Vocabulary
import random
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import retrieval_post
from retrieval_post import showurl
from retrieval_post import mdict
from retrieval_post import filelist
from retrieval_post import action
from retrieval_post import caption_model
# pkill -f uwsgi -9
# uwsgi --ini /root/server/config.ini --daemonize log/uwsgi.log
app = Flask(__name__)


@app.route('/')
def index():
    return 'hello dhhddd!!!'
@app.route('/retrieval_new', methods=['POST', 'GET'])
def retrieval_new():
    queryImage = request.values.get('queryImage')
    num_result = request.values.get('num_result', default=5)
    queryText = request.values.get('queryText', default="")
    if (queryImage == "" and queryText == ""):
        return "invalid input"
    elif (queryImage == ""):
        query_caption, result_captions, result_images, type = retrieval_post.Text_retrieval_only(queryText, num_result)
    elif (queryText == ""):
        query_caption, result_captions, result_images, type = retrieval_post.Image_retrieval_only(queryImage, num_result)
    else:
        query_caption, result_captions, result_images, type = retrieval_post.Text_Image_Retrieval(queryImage,queryText,num_result)
    resSets = retrieval_post.makeResponse(query_caption, result_captions, result_images, type)
    return resSets


@app.route('/get_querys', methods=['POST', 'GET'])
def recognition():
    resSets = {}
    resSets["result_images"] = []
    resSets["result_captions"] = []
    num_result = request.values.get('num_result', default=5)
    try:
        choic = random.sample(filelist, int(num_result))
    except ValueError:
        choic = filelist
    for i in choic:
        resSets["result_images"].append(showurl + i)
        # resSets["result_captions"].append(caption_model.generate_caption(i.split('/')[-1], num_result, False,True))
    return resSets


@app.route('/image_show', methods=['POST', 'GET'])
def imageShow():
    imgPath = request.values.get('path', default='retrievalDB/oxford_001189.jpg')
    try:
        mime = mdict[(imgPath.split('/')[-1]).split('.')[-1]]
        with open(imgPath, 'rb') as f:
            image = f.read()

        return Response(image, mimetype=mime)
    except:
        return "not real path"


if __name__ == '__main__':
    app.debug = False
    app.run(host="0.0.0.0", threaded=True, port='6009')
