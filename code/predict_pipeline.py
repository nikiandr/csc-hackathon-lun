import lightgbm as lgb
import numpy as np
from PIL import Image, ImageChops
import cv2
import imagehash
import pandas as pd
import joblib
from time import time


class PredictPipeline:
    def __init__(self, model_path='./lgb_classifier_v2.9_sift.pkl', sigma=2.0, contrast_threshold=0.01, edge_threshold=20):
        self.orb = cv2.ORB_create()
        FLANN_INDEX_LSH = 6
        index_params = dict(
            algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1
        )
        search_params = dict(checks=50)
        self.flannORB = cv2.FlannBasedMatcher(index_params, search_params)

        self.sift = cv2.SIFT_create(sigma=sigma, contrastThreshold=contrast_threshold, edgeThreshold=edge_threshold)
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        self.flannSIFT = cv2.FlannBasedMatcher(index_params, search_params)

        self.background_threshold = 20
        self.hash_funcs = {
            "ahash_8": lambda x: imagehash.average_hash(x, 8),
            "ahash_4": lambda x: imagehash.average_hash(x, 4),
            "phash_8": lambda x: imagehash.phash(x, hash_size=8),
            "phash_4": lambda x: imagehash.phash(x, hash_size=4),
            "dhash_4": lambda x: imagehash.dhash(x, hash_size=4),
            "dhash_8": lambda x: imagehash.dhash(x, hash_size=8),
            "colorhash_21": lambda x: imagehash.colorhash(x, binbits=21),
            "whash_4_haar": lambda x: imagehash.whash(x, 4, mode="haar"),
            "whash_8_haar": lambda x: imagehash.whash(x, 8, mode="haar"),
        }
        self.features_count = 18
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

            matches = self.flannORB.knnMatch(des1, des2, k=2)

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
        
    def sift_similarity(self, img1, img2):
        try:
            kp1, des1 = self.sift.detectAndCompute(img1,None)
            kp2, des2 = self.sift.detectAndCompute(img2,None)

            matches = self.flannSIFT.knnMatch(des1, des2, k=2)

            good_matches_count = 0
            for m, n in matches:
                if m.distance < 0.7*n.distance:
                    good_matches_count += 1

            similarity = good_matches_count/len(kp2)
            return similarity
        except Exception:
            return 0
        
    def get_hashres(self, img):
        return {k: transform(img) for k, transform in self.hash_funcs.items()}

    def predict_proba_open(self, image_path1, image_path2):
        img1 = Image.open(image_path1)
        img2 = Image.open(image_path2)
        return self.predict_proba(img1, img2)

    def predict_proba(self, img1, img2):
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
        X_test = pd.DataFrame(columns=["ahash_4", "ahash_8", "colorhash_21", "dhash_4", "dhash_8",
                                       "height_diff", "height_ratio", "left_height", "left_width",
                                        "phash_4", "phash_8", "right_height", "right_width", "sift_similarity", 
                                        "whash_4_haar", "whash_8_haar", "width_diff", "width_ratio"], 
                                        data=[[0] *  self.features_count])
        for k in hashres:
            X_test.loc[0, k] = hashres[k]

        img1_cv = np.array(img1.convert('L'))
        img2_cv = np.array(img2.convert('L'))
        # X_test.loc[0, "orb_similarity"] = self.orb_similarity(img1_cv, img2_cv)
        X_test.loc[0, "sift_similarity"] = self.sift_similarity(img1_cv, img2_cv)

        X_test.loc[0, "left_width"], X_test.loc[0, "left_height"] = img1.size
        X_test.loc[0, "right_width"], X_test.loc[0, "right_height"] = img2.size
        X_test["width_diff"] = np.abs(X_test["left_width"] - X_test["right_width"])
        X_test["height_diff"] = np.abs(X_test["left_height"] - X_test["right_height"])
        X_test["width_ratio"] = X_test.apply(lambda row: row["left_width"] / row["right_width"] if row["right_width"] != 0 else 0, axis=1)
        X_test["height_ratio"] = X_test.apply(lambda row: row["left_height"] / row["right_height"] if row["right_width"] != 0 else 0, axis=1)

        # print(X_test)

        res = self.model.predict_proba(X_test, num_threads=1)[0, 1]
        return res, X_test


if __name__ == '__main__':
    pipeline = PredictPipeline()
    # res = pipeline.predict_proba_open('../dataset/images_test/891059851.jpg', '../dataset/images_test/925809022.jpg')
    # print(res)
    df_test = pd.read_csv('../dataset/test_df_with_features_v1.csv')
    image_pairs = []
    test_path = '../dataset/images_test/'
    collected = 0
    for i, row in df_test.iterrows():
        try:
            image_pairs.append((Image.open(test_path + row["image_url1"]), Image.open(test_path + row["image_url2"])))
            collected += 1
            if collected > 100:
                break
        except:
            pass
    broken = 0
    # cv2.setNumThreads(1)
    start_time = time()
    for pair in image_pairs:
        try:
            pipeline.predict_proba(pair[0], pair[1])
        except:
            broken += 1
    print(f"Broken count = {broken}")
    print(f"Exec time = {time() - start_time}")
