#! -*- coding: utf-8 -*-

"""
  [model_tf]
    python model_tf.py --help
"""

#---------------------------------
# モジュールのインポート
#---------------------------------
import tensorflow as tf

#---------------------------------
# 定数定義
#---------------------------------

#---------------------------------
# 関数
#---------------------------------

#---------------------------------
# クラス
#---------------------------------
class TensorFlowModel():
	# --- コンストラクタ ---
	def __init__(self, flag_tflite=False):
		self.flag_tflite = flag_tflite
		if not self.flag_tflite:
			# TensorFlowモデルを読み込む場合の処理
			self.sess = tf.Session()
		else:
			# TensorFlow Liteモデルを読み込む場合の処理
			self.sess = tf.Session()
			pass

		return

	# --- 学習済みモデルの読み込み ---
	def load_model(self, trained_model):
		if not self.flag_tflite:
			# TensorFlowモデルを読み込む場合の処理
			saver = tf.train.import_meta_graph(trained_model + '.ckpt.meta', clear_devices=True)
			saver.restore(self.sess, trained_model + '.ckpt')
			gd = tf.GraphDef.FromString(open(trained_model + '_frozen.pb', 'rb').read())
			self.x, self.y = tf.import_graph_def(gd, return_elements=['input:0', 'MobilenetV1/Predictions/Reshape_1:0'])	# [T.B.D]
		else:
			# TensorFlow Liteモデルを読み込む場合の処理
			self.interpreter = tf.lite.Interpreter(model_path=trained_model)
			self.interpreter.allocate_tensors()
			self.input_details = self.interpreter.get_input_details()
			self.output_details = self.interpreter.get_output_details()

		print('[INFO] model is restored')

		return

	# --- 推論 ---
	def inference(self, data):
		if not self.flag_tflite:
			# TensorFlowモデルを読み込む場合の処理
			prediction = self.sess.run(self.y, feed_dict={self.x: data})
		else:
			# TensorFlow Liteモデルを読み込む場合の処理
			input_shape = self.input_details[0]['shape']
			self.interpreter.set_tensor(self.input_details[0]['index'], data)
			self.interpreter.invoke()
			prediction = self.interpreter.get_tensor(self.output_details[0]['index'])

		return prediction

#---------------------------------
# メイン処理
#---------------------------------
if __name__ == '__main__':
	# --- import module ---
	import sys
	import argparse
	import numpy as np
	import pandas as pd
	from collections import OrderedDict
	from data_loader import DataLoader

	# --- local functions ---
	"""
	  関数名: _arg_parser
	  説明：引数を解析して値を取得する
	"""
	def _arg_parser():
		parser = argparse.ArgumentParser(description='TensorFlowによる学習・推論', formatter_class=argparse.RawTextHelpFormatter)

		# --- 引数を追加 ---
		parser.add_argument('--mode', dest='mode', type=int, default=0, help='0:推論モード(推論のみ，ラベルとの比較なし)\n1:学習モード,\n2:テストモード(推論と正解ラベルとの比較)', required=True)
		parser.add_argument('--train_csv', dest='train_csv', type=str, default=None, help='学習データのリスト\n[csv構造]\n  image path, label', required=False)
		parser.add_argument('--test_csv', dest='test_csv', type=str, default=None, help='テストデータのリスト\n[csv構造]\n  image path, label', required=False)
		parser.add_argument('--inference_csv', dest='inference_csv', type=str, default=None, help='推論データのリスト\n[csv構造]\n  image path', required=False)
		parser.add_argument('--readable_names_csv', dest='readable_names_csv', type=str, default=None, help='推論クラスとクラス名のマッピングリスト', required=False)
		parser.add_argument('--trained_model', dest='trained_model', type=str, default=None, help='学習済みモデル(例: mobilenet_v1_1.0_224)', required=False)

		args = parser.parse_args()

		return args

	# --- 引数処理 ---
	args = _arg_parser()

	# --- TensorFlowモデルオブジェクト構築 ---
	model = TensorFlowModel()

	# --- モードに応じて処理 ---
	if (args.mode == 0):
		# --- 推論モード ---
		if (args.trained_model is None):
			sys.stderr.write('[ERROR] model_path is not indicated\n')
			quit()

		# --- 推論クラスとクラス名をマッピング ---
		if (args.readable_names_csv is not None):
			od_class_name = OrderedDict(pd.read_csv(args.readable_names_csv, header=None).values)

		# --- データ読み込み ---
		dl = DataLoader(args.inference_csv, 224, 224)   # (224, 224): MobileNetで学習済みの画像サイズ(ImageNet)
		images, _ = dl.load_data()

		# --- モデル読み込み ---
		model.load_model(args.trained_model)

		# --- 推論 ---
		images = (images / 128) - 1.0
		predict = model.inference(images)
		predict_class = np.argmax(predict, axis=1)

		if (args.readable_names_csv is not None):
			for _class in predict_class:
				print('{}: {}'.format(_class, od_class_name[_class]))
		else:
			print(predict)

