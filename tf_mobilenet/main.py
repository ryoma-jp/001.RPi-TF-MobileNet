#! -*- coding: utf-8 -*-

"""
  [main]
    python main.py --help
"""

#---------------------------------
# モジュールのインポート
#---------------------------------
import cv2
import time
import argparse
import numpy as np
import pandas as pd
from collections import OrderedDict
from data_loader import DataLoader
from model_tf import TensorFlowModel

#---------------------------------
# 定数定義
#---------------------------------
CLASS_NAMES = 'readable_names_for_imagenet_label.csv'

#---------------------------------
# 関数
#---------------------------------
def ArgParser():
	parser = argparse.ArgumentParser(description='TensorFlowによる学習・推論', formatter_class=argparse.RawTextHelpFormatter)

	# --- 引数を追加 ---
	parser.add_argument('--mode', dest='mode', type=int, default=None, help='動作モード(0: カメラモード, 1: ファイル読み込みモード)', required=True)
	parser.add_argument('--trained_model', dest='trained_model', type=str, default=None, help='学習済みモデル(例: mobilenet_v1_1.0_224)', required=True)
	parser.add_argument('--inference_csv', dest='inference_csv', type=str, default=None, help='推論データのリスト\n[csv構造]\n  image path', required=False)

	args = parser.parse_args()

	return args

def main():
	# --- 引数処理 ---
	args = ArgParser()

	# --- TensorFlowモデルオブジェクト構築 ---
	model = TensorFlowModel()
	model.load_model(args.trained_model)

	# --- モードに応じて処理 ---
	od_class_name = OrderedDict(pd.read_csv(CLASS_NAMES, header=None).values)
	if (args.mode == 0):
		# --- Camera mode ---

		# --- Camera settings ---
		cap = cv2.VideoCapture(0)
		cap.set(cv2.CAP_PROP_FPS, 30)
		cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
		cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

		# --- Capture loop ---
		while (True):
			ret, frame = cap.read()
			cv2.imshow('frame', frame)
	
			if (cv2.waitKey(1) & 0xFF == ord('q')):
				break
	
			# --- inference image ---
			img = np.array([(cv2.resize(frame, (224, 224)) / 128) - 1.0])
			start = time.time()
			predict = model.inference(img)
			end = time.time()
			predict_class = np.argmax(predict[0])
			print('{}, {:3.2}fps: {}'.format(predict_class, 1/(end-start), od_class_name[predict_class]))
	
		cap.release()
		cv2.destroyAllWindows()

	else:
		# --- file mode ---

		# --- データ読み込み ---
		dl = DataLoader(args.inference_csv, 224, 224)   # (224, 224): MobileNetで学習済みの画像サイズ(ImageNet)
		images, _ = dl.load_data()

		# --- 推論 ---
		images = (images / 128) - 1.0
		predict = model.inference(images)
		predict_class = np.argmax(predict, axis=1)

		for _class in predict_class:
			print('{}: {}'.format(_class, od_class_name[_class]))

#---------------------------------
# メイン処理
#---------------------------------
if __name__ == '__main__':
	main()

