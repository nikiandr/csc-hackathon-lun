import lightgbm as lgb
import numpy as np
from PIL import Image, ImageChops
import cv2
import imagehash
import pandas as pd
import joblib


class PredictPipeline:
    def __init__(self, model_path='./lgb_classifier_v2.9_med_features.pkl'):
        self.orb = cv2.ORB_create()
        FLANN_INDEX_LSH = 6
        index_params = dict(
            algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1
        )
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        self.background_threshold = 20
        self.hash_funcs = {
            "ahash_8": lambda x: imagehash.average_hash(x, 8),
            "phash_8": lambda x: imagehash.phash(x, hash_size=8),
            "dhash_8": lambda x: imagehash.dhash(x, hash_size=8),
            "colorhash_21": lambda x: imagehash.colorhash(x, binbits=21),
            "whash_8_haar": lambda x: imagehash.whash(x, 8, mode="haar"),
        }
        self.model = joblib.load(model_path)


    def remove_padding(self, image):
        mode = image.mode
        image = image.convert('RGB')
        background_color_1 = image.getpixel((0,0))
        background = Image.new(image.mode, image.size, background_color_1)
        difference = ImageChops.difference(image, background)
        difference = ImageChops.add(difference, difference, 2.0, -self.background_threshold)
        bbox = difference.getbbox()
        if bbox:
            image = image.crop(bbox)
        w, h = image.size
        background_color_2 = image.getpixel((w-1, h-1))
        background = Image.new(image.mode, image.size, background_color_2)
        difference = ImageChops.difference(image, background)
        difference = ImageChops.add(difference, difference, 2.0, -self.background_threshold)
        bbox = difference.getbbox()
        if bbox:
            image = image.crop(bbox)
        return image.convert(mode)
    
    def orb_similarity(self, img1, img2):
        try:
            kp1, des1 = self.orb.detectAndCompute(img1, None)
            kp2, des2 = self.orb.detectAndCompute(img2, None)

            matches = self.flann.knnMatch(des1, des2, k=2)

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
        except Exception:
            return None
        
    def get_hashres(self, img):
        return {k: transform(img) for k, transform in self.hash_funcs.items()}

    def predict_proba(self, image_path1, image_path2):
        img1 = Image.open(image_path1)
        img2 = Image.open(image_path2)
        img1 = self.remove_padding(img1)
        img2 = self.remove_padding(img2)
        hashes1 = self.get_hashres(img1)
        hashes2 = self.get_hashres(img2)
        hashres = {}
        for k in hashes1:
            try:
                hashres[k] = hashes1[k] - hashes2[k]
            except:
                hashres[k] = np.nan
        X_test = pd.DataFrame(columns=["ahash_8", "colorhash_21", "dhash_8", "phash_8", "whash_8_haar", "orb_similarity"], data=[[0] * 6])
        for k in hashres:
            X_test.loc[0, k] = hashres[k]

        img1_cv = np.array(img1.convert('L'))
        img2_cv = np.array(img2.convert('L'))
        X_test.loc[0, "orb_similarity"] = self.orb_similarity(img1_cv, img2_cv)

        res = self.model.predict_proba(X_test)[0, 1]
        return(res)

if __name__ == '__main__':
    pipeline = PredictPipeline()
    print(pipeline.predict_proba('../dataset/images_test/891059851.jpg', '../dataset/images_test/925809022.jpg'))