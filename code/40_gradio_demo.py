import gradio as gr


from PIL import Image, ImageChops
import cv2
import imagehash
import pandas as pd
import numpy as np

THRESHOLD = 20

# cv2.setNumThreads(1)

clf = pd.read_pickle("../models/lgbm_v2.9_orb.pkl")

def remove_padding(image):
    mode = image.mode
    image = image.convert("RGB")
    background_color_1 = image.getpixel((0, 0))
    background = Image.new(image.mode, image.size, background_color_1)
    difference = ImageChops.difference(image, background)
    difference = ImageChops.add(difference, difference, 2.0, -THRESHOLD)
    bbox = difference.getbbox()
    if bbox:
        image = image.crop(bbox)
    w, h = image.size
    background_color_2 = image.getpixel((w - 1, h - 1))
    background = Image.new(image.mode, image.size, background_color_2)
    difference = ImageChops.difference(image, background)
    difference = ImageChops.add(difference, difference, 2.0, -THRESHOLD)
    bbox = difference.getbbox()
    if bbox:
        image = image.crop(bbox)
    return image.convert(mode)


def orb_similarity(img1, img2):
    orb = cv2.ORB_create()
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    try:
        img1 = np.array(img1.convert("L"))
        img2 = np.array(img2.convert("L"))
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        matches = flann.knnMatch(des1, des2, k=2)

        good_matches_count = 0
        for pair in matches:
            try:
                m, n = pair
                if m.distance < 0.7 * n.distance:
                    good_matches_count += 1

            except ValueError:
                pass

        similarity = 2 * good_matches_count / (len(kp1) + len(kp2))
        return similarity
    except KeyboardInterrupt:
        raise
    # except Exception:
    #     return None


def get_features(img1, img2):
    return {
        "unpadded.ahash_8": imagehash.average_hash(img1, hash_size=8) - imagehash.average_hash(img2, hash_size=8),
        "unpadded.colorhash_21": imagehash.colorhash(img1, binbits=21) - imagehash.colorhash(img2, binbits=21),
        "unpadded.dhash_8": imagehash.dhash(img1, hash_size=8) - imagehash.dhash(img2, hash_size=8),
        "unpadded.phash_8": imagehash.phash(img1, hash_size=8) - imagehash.phash(img2, hash_size=8),
        "unpadded.orb_similarity": orb_similarity(img1, img2),
        "unpadded.whash_8_haar": imagehash.whash(img1, 8, mode="haar") - imagehash.whash(img2, 8, mode="haar"),
    }


def predict(model, img1, img2):
    img1 = remove_padding(img1)
    img2 = remove_padding(img2)
    feats = get_features(img1, img2)
    return (pd.DataFrame(feats, index=[0]),
            model.predict(pd.DataFrame(feats, index=[0]), num_threads=1))
    # print(feats)
    # return model.predict(pd.DataFrame(feats, index=[0]), num_threads=1)


def greet(img1, img2):
    img1_pil = Image.fromarray(img1).convert('RGB')
    img2_pil = Image.fromarray(img2).convert('RGB')
    print(predict(clf, img1_pil, img2_pil)[0])
    print(predict(clf, img1_pil, img2_pil)[0].columns.values)
    print(predict(clf, img1_pil, img2_pil)[1])
    features, prd = predict(clf, img1_pil, img2_pil)
    features.columns = ['ahash_8', 'colorhash_21', 'dhash_8', 'phash_8', 'orb_similarity', 'whash_8_haar']
    print(features)
    return features, prd[0]
    # return "a"


# demo = gr.Interface(fn=greet, 
#                     inputs=["image", "image"], 
#                     outputs="label")

demo = gr.Interface(fn=greet, inputs=["image", "image"], outputs=[gr.Dataframe(row_count=(1, "dynamic"),
                                                                               col_count=(6, "dynamic"),
                                                                               label="Features",
                                                                               headers= ['ahash_8',
                                                                               'colorhash_21', 
                                                                               'dhash_8', 
                                                                               'phash_8', 
                                                                               'orb_similarity', 
                                                                               'whash_8_haar']),
                                                                  "label"])


demo.launch()
