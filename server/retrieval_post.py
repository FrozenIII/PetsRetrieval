import base64
from io import BytesIO
import random
import torch.multiprocessing as mp
# mp.set_start_method('spawn')
from image_search import RetrievalModel
from ImageCaption import ImageCaption
from PIL import Image
from TextSearch import MDBSearch
from ImageCaption import pets_cap
showurl="https://dhhddd.mynatapp.cc/image_show?path="
mdict = {
        'jpeg': 'image/jpeg',
        'jpg': 'image/jpeg',
        'png': 'image/png',
        'gif': 'image/gif'
    }
action=["laughing", "playing", "moving", "happy", "sad"]

caption_model = ImageCaption()
retrieval_model = RetrievalModel()
filelist = retrieval_model.getqimages()
searchText = MDBSearch()
searchText.clear()
searchText.create_table(retrieval_model.images, pets_cap, type=2)
def Image_retrieval_only(queryImage,num_result):
    if queryImage.startswith(showurl):
        image_data = queryImage[len(showurl):]
        query_caption=caption_model.generate_caption(image_data, num_result, False, True)
        result_images, _ = retrieval_model.retrieval_image(image_data, query_caption, num_result, False)
        type = 0
    else:
        try:
            byte_data = base64.b64decode(queryImage)
        except:
            print("invalid query image")
            return{"text_descriptor" : "invalid query image"}
            return resSets
        image_data = BytesIO(byte_data)
        #             try:
        img = Image.open(image_data)
        query_caption = caption_model.generate_caption(img, num_result, True)
        result_images, type = retrieval_model.retrieval_image(img, query_caption, num_result, True)
    if type!= 1:
        result_captions=[]
        for i in result_images:
            result_captions.append(caption_model.generate_caption(i.split('/')[-1], num_result, False,True))
    else:
        result_captions = []
    return query_caption,result_captions,result_images,type

def Text_retrieval_only(queryText, num_result=20, forImage=False):
    f_res = searchText.search(queryText)  # Output is a set!!! Precisely search for each word
    # result_list_id=searchText.return_top_k(queryText, f_res, k=20, thres=0.8)
    try:
        if forImage:
            result_list_id = random.sample(f_res, 500)
        else:
            result_list_id = random.sample(f_res, 50)

    except ValueError:
        result_list_id = f_res
    result_list = [retrieval_model.images[i] for i in result_list_id]
    result_captions=[]
    for i in result_list:
        result_captions.append(caption_model.generate_caption(i.split('/')[-1], num_result, False,True))
    return "",result_captions,result_list,0


def Text_Image_Retrieval(queryImage, queryText, num_result):
    query_caption,result_captions,result_images,type=Image_retrieval_only(queryImage, num_result)
    _,_,result_images_t,_ = Text_retrieval_only(queryText, num_result=500, forImage=True)
    real_result_images=[]
    real_result_captions=[]
    for idx,i in enumerate(result_images):
        if i in result_images_t:
            real_result_images.append(i)
            real_result_captions.append(result_captions[idx])
    return query_caption,real_result_captions,real_result_images,type

def makeResponse(query_caption,result_captions,result_images,type):
    resSets = {}
    resSets["query_caption"]=query_caption
    resSets["result_captions"] = result_captions
    resSets["result_images"] = [showurl + i for i in result_images]
    resSets["type"] = type
    return resSets