#! -*- coding: utf-8 -*-

"""
  [data_loader]
		python data_loader.py --help
"""

#---------------------------------
# モジュールのインポート
#---------------------------------
import os
import sys
import cv2
import argparse
import numpy as np
import pandas as pd

#---------------------------------
# 定数定義
#---------------------------------

#---------------------------------
# 関数
#---------------------------------

#---------------------------------
# クラス
#---------------------------------
"""
	データを読み込むクラス
	csvファイルのヘッダは下記に対応
	  image file: 画像データとして読み込む
	  label: 正解ラベル
"""
class DataLoader():
	# --- コンストラクタ ---
	def __init__(self, data_csv, img_h=-1, img_w=-1):
		# --- csvファイルをpandasで読み込む ---
		self.df_data = pd.read_csv(data_csv)
#		print(self.df_data)
#		print(self.df_data.loc[:'image file'])
#		print(self.df_data.columns)

		# --- その他のパラメータ取り込み ---
		self.img_h = img_h
		self.img_w = img_w

		return

	# --- 画像データ読み込み ---
	def load_images(self, data):
		print(data)
		idx = 0
		img = None
		while (img is None):
			img = cv2.imread(data[idx][0])
			idx += 1

		if ((self.img_h > 0) and (self.img_w > 0)):
			img = cv2.resize(img, (self.img_h, self.img_w))
			cv2.imwrite('./debug_resize.png', img)
		images = np.array([img])
		for _data in data[idx:]:
			img = cv2.imread(_data[0])
			if ((self.img_h > 0) and (self.img_w > 0)):
				img = cv2.resize(img, (self.img_h, self.img_w))
			if (img is not None):
				images = np.vstack((images, np.array([img])))

		return images

	# --- データ読み込み ---
	def load_data(self):
		data = None
		label = None
		for data_type in self.df_data.columns:
			if (data_type == 'image file'):
				# 画像データを読み込む
				data = self.load_images(self.df_data.loc[:data_type].values)

			elif (data_type == 'label'):
				# ラベルデータを読み込む
				pass	# [T.B.D]
			else:
				# 未対応のデータ形式
				sys.stderr.write('[ERROR] unknown data type: {}'.format(data_type))
				return None, None

		return (data, label)


#---------------------------------
# メイン処理
#---------------------------------
if __name__ == '__main__':
	"""
	  関数名: _arg_parser
	  説明：引数を解析して値を取得する
	"""
	def _arg_parser():
		parser = argparse.ArgumentParser(description='csv形式で記述されたデータリストをnumpy配列に変換する', formatter_class=argparse.RawTextHelpFormatter)
	
		# --- 引数を追加 ---
		parser.add_argument('--data_csv', dest='data_csv', type=str, default=None, help='データのリスト\n[csv構造]\n  image path(, label)', required=True)
		parser.add_argument('--img_h', dest='img_h', type=int, default=-1, help='読み込んだ画像データのリサイズ後の水平サイズ', required=False)
		parser.add_argument('--img_w', dest='img_w', type=int, default=-1, help='読み込んだ画像データのリサイズ後の垂直サイズ', required=False)
	
		args = parser.parse_args()
	
		return args

	# --- 引数処理 ---
	args = _arg_parser()

	# --- データ読み込みオブジェクト構築 ---
	dl = DataLoader(args.data_csv, args.img_h, args.img_w)
	loaded_data = dl.load_data()

	print(loaded_data)
	print(loaded_data[0].shape)

