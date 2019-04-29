# 001.RPi-TF-MobileNet: Raspberry Pi上でTensorFlowのMobileNetを動かす

## 環境

* RaspberryPi 3 Model B+
* TensorFlow 1.13.1
	* MobileNet_v1_1.0_224  
https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md  
http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224.tgz

## 手順

### Raspberry Piの準備

1. Raspbian Stretch with desktopをダウンロード  
ダウンロード後，解凍する  
https://www.raspberrypi.org/downloads/raspbian/
1. SDカードをSD Card FormatterでFAT32でフォーマットする  
https://www.sdcard.org/jp/downloads/formatter/
1. SDカードにRaspbian Stretch with desktopを書き込む  
Win32 Disc Imagerでimgファイルを指定してWriteする  
https://forest.watch.impress.co.jp/docs/review/1067836.html
1. Raspberry Piの初期設定を行う(sudo raspi-config)  
下記設定後，再起動するかを聞かれるので，Yesを選択して再起動する  
	* raspi-configの更新  
	→「8 Update」を選択
	* ロケールの設定  
	→「4 Localisation Options」を選択
		* I1 Change Locale：ja_JP.UTF-8 UTF-8
		* I2 Change Timezone：Asia → Tokyoを選択
		* I3 Change Keyboard Layout：Generic 105-key (Intl) PC → Other → Japanese → Japanese (OADG 109A) → The default for the keyboard layout → No compose keyを選択
		* I4 Change Wi-fi Country：JP Japanを選択
	* SSH serverの有効化  
	→「5 Interfacing Options」→「P2 SSH」→「Yes」を選択し，piユーザのパスワードを変更する
	* ファイルシステムの拡張  
	→「7 Advanced Options」→「A1 Expand Filesystem」を選択
1. パッケージ，OS，ファームウェアの更新

		$ sudo apt-get update
		$ sudo apt-get upgrade
		$ sudo apt-get dist-upgrade
		$ sudo apt-get install rpi-update
		$ sudo rpi-update
		$ sudo reboot

### TensorFlowインストール

仮想環境(virtualenv)を構築して，TensorFlowをインストールする．  
https://www.tensorflow.org/install/pip

	$ sudo apt-get install python3-dev python3-pip
	$ sudo apt-get install libatlas-base-dev
	$ sudo pip3 install -U virtualenv
	$ virtualenv --system-site-packages -p python3 ./tensorflow
	$ source ./tensorflow/bin/activate
	(tensorflow) $ pip install --upgrade pip
	(tensorflow) $ pip list
	(tensorflow) $ pip install --upgrade tensorflow
	(tensorflow) $ python -c "import tensorflow as tf; tf.enable_eager_execution(); print(tf.reduce_sum(tf.random_normal([1000, 1000])))"   # Verify the install
	(tensorflow) $ deactivate

### MobileNetの実行

画像を処理するのでOpenCVをインストールする  

	$ curl -SL https://github.com/mt08xx/files/raw/master/opencv-rpi/libopencv3_3.4.6-20190415.1_armhf.deb -o libopencv3_3.4.6-20190415.1_armhf.deb
	$ sudo apt-get autoremove -y libopencv3
	$ sudo apt-get install -y ./libopencv3_3.4.6-20190415.1_armhf.deb

MobileNetの学習済みモデルをダウンロード

	$ mkdir models
	$ cd models
	$ wget http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224.tgz
	$ tar -zxf mobilenet_v1_1.0_224.tgz
	$ cd ..
	
	
#### 画像ファイルを用いた推論

	$ python model_tf.py --mode 0 --trained_model models/mobilenet_v1_1.0_224 --inference_csv ./inference.csv --readable_names_csv ./readable_names_for_imagenet_label.csv

--inference_csvで推論する画像ファイルを指定

	$ cat inference.csv
	image file
	/home/pi/work/tensorflow/work/images/14720420_8830007ef2.jpg
	/home/pi/work/tensorflow/work/images/20170612223613.jpg
	/home/pi/work/tensorflow/work/images/main_232393_14718_detail.jpg

--readable_names_csvで推論クラスとクラス名をマッピング

	$ head readable_names_for_imagenet_label.csv
	1,"tench, Tinca tinca"
	2,"goldfish, Carassius auratus"
	3,"great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias"
	4,"tiger shark, Galeocerdo cuvieri"
	5,"hammerhead, hammerhead shark"
	6,"electric ray, crampfish, numbfish, torpedo"
	7,stingray
	8,cock
	9,hen
	10,"ostrich, Struthio camelus"

#### ラズパイカメラで撮影した画像の推論

## 参考URL

|title|URL|
|:----|:----|
|MobileNet_v1_1.0_224|https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md|
|Raspbian|https://www.raspberrypi.org/downloads/raspbian/|
|SD Card Formatter|https://www.sdcard.org/jp/downloads/formatter/|
|Win32 Disc Imager|https://forest.watch.impress.co.jp/docs/review/1067836.html|
|TensorFlow install|https://www.tensorflow.org/install/pip|
|tensorflow_data_extractor - mismatch between the current graph and the graph from the checkpoint . Cannot assign a device for operation 'parallel_read/filenames/Greater'|https://github.com/ARM-software/ComputeLibrary/issues/504|
|ラズパイ3にOpenCV3/4を簡単に導入|https://qiita.com/mt08/items/e8e8e728cf106ac83218|

## その他

### vimタブ設定

/usr/share/vim/vim80/ftplugin/以下のpython.vimを修正して，Tabキー入力時の文字をタブ 4文字分に変更  

	$ diff /usr/share/vim/vim80/ftplugin/python.vim.org /usr/share/vim/vim80/ftplugin/python.vim
	71c71,72
	<     setlocal expandtab shiftwidth=4 softtabstop=4 tabstop=8
	---
	>     " setlocal expandtab shiftwidth=4 softtabstop=4 tabstop=8
	>     setlocal noexpandtab shiftwidth=4 softtabstop=4 tabstop=4

