import os
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.cluster import KMeans
from glob import glob
import shutil

#---------------------------------------------------------
# tensorflow2.x + kerasでGPUメモリの使用量を抑える処理
#---------------------------------------------------------
# GPUデバイスのリストを取得
gpus = tf.config.experimental.list_physical_devices('GPU')
# GPUがあれば設定する
if gpus:
  try:
    # GPUメモリの使用量を制限する
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # 設定に失敗した時の例外エラー
    print(e)

#---------------------------------------------------------
# フォルダにある画像読込処理
#---------------------------------------------------------
# imgフォルダにある画像読込
impathlist = glob('img/*.jpg')
impathlist.extend(glob('img/*.jpeg'))
impathlist.extend(glob('img/*.png'))
impathlist.extend(glob('img/*.bmp'))

if (len(impathlist)==0):
  print("フォルダに画像がありません")
  exit()

imlist = []
for p in impathlist:
    # kerasのpreprocessing.imageで画像読込
    img = tf.keras.preprocessing.image.load_img(p, target_size=(299, 299))
    # kerasのpreprocessing.imageで配列に変換
    x = tf.keras.preprocessing.image.img_to_array(img)
    # inception_v3で読み込めるテンソル配列に変換
    x = tf.keras.applications.inception_v3.preprocess_input(x)
    # リストに追加
    imlist.append(x)
# numpy配列に変換
imlist = np.array(imlist)
print("画像読込完了")

#---------------------------------------------------------
# ボトルネック特徴量取得処理
#---------------------------------------------------------
# InceptionV3のモデル生成
model = tf.keras.applications.InceptionV3(weights='imagenet', input_shape=[299, 299, 3],include_top=False)
# 推論
bottleneck_features = model.predict(imlist)
# InceptionV3でボトルネック特徴量取得(出力レイヤーの1つ手前のレイヤーを取得)
bottleneck_features = bottleneck_features.reshape(bottleneck_features.shape[0], -1)


#---------------------------------------------------------
#クラスタリング処理
#---------------------------------------------------------
# クラスタ数設定
n_clusters = 4
print("クラスタ数設定:" + str(n_clusters))

#KMeansでクラスタリング
model = KMeans(n_clusters=n_clusters,n_jobs=-1,tol=0.0001,n_init=10,max_iter=300,init='k-means++').fit(bottleneck_features)

# 学習結果のラベル（画像ファイル名）取得
labels = model.labels_
print("K-means クラスタリング完了")


#---------------------------------------------------------
#クラスタリング結果をラベルごとにフォルダ分け
#---------------------------------------------------------
#前回フォルダを削除
if os.path.exists('./output/'):
   shutil.rmtree('./output/')
   print("outputフォルダ削除完了")

# クラスタリング結果をラベルごとにフォルダ分け
output_path = "./output"
for i in range(n_clusters):
    label = np.where(labels==i)[0]
    # Image placing
    if not os.path.exists(output_path+"/img_"+str(i)):
        os.makedirs(output_path+"/img_"+str(i))

    for j in label:
        img = Image.open(impathlist[j])
        fname = impathlist[j].split('\\')[-1]
        img.save(output_path+"/img_"+str(i)+"/" + fname)

print("画像のフォルダ分け完了")
