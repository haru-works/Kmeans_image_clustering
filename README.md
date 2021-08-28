# 似た画像でフォルダ分けするスクリプト
K-meansを使ってフォルダ内の似た画像をリストアップしてフォルダ分けするPythonスクリプトです。
<br>
画像の特徴量抽出には、KerasのInceptionV3を使っています。
<br>
Windows10 Pro上で作ったので、Windows10 Proしか動作保証しません。
<br>
GPUがなくてもCPUで動きます。（処理は遅くなると思いますが・・・）

## 環境構築手順

### pythonの準備

pythonの公式(https://www.python.org/downloads/windows/) からPython3を落としてインストールします。
<br>
※TensorFlowのPython サポート状況
<br>
Python 3.9 入れちゃった場合、TensorFlow 2.5 以降をインストールしてね。
<br>
Python 3.8 入れちゃった場合、TensorFlow 2.2 以降をインストールしてね。
<br>

### 仮想環境の準備

venvをつかって仮想環境を作ります。
<br>
作り方は、下記のサイトを参考に仮想環境を作って下さい。
<br>
https://qiita.com/fiftystorm36/items/b2fd47cf32c7694adc2e
<br>
<br>
仮想環境が出来たらActivateします。
<br>
image-clustering.pyがあるフォルダに移動します。
<br>
下記のコマンドで、requirements.txtに書かれた必要なライブラリを入れます。
```
pip install -r requirements.txt
```

### GPU環境の準備

NVIDIA系のGPUがあるPCの場合、NVIDIA CUDAを設定すると処理が早くなります。
<br>
requirements.txtに書かれているTensorFlowのバージョンは2.6なので、NVIDIA CUDA ツールキット 11.3，NVIDIA cuDNN 8.2でTensorFlow-GPUが動くようです。
<br>
下記のサイトを参考に入れてみてください。
<br>
https://www.kkaneko.jp/tools/win/keras.html#S1
<br>

### プログラム実行

１．imgフォルダに適当な画像を入れます。
<br>
２．image-clustering.pyがあるフォルダに移動します。
<br>
３．下記のコマンドを実行します。
<br>
```
python image-clustering.py
```
４．処理に成功すると、outputフォルダが作られて、img_0～img_xxのクラスタ数分(初期値：n_clusters = 4)のフォルダが作られて、似た画像同士でフォルダ分けされます。

