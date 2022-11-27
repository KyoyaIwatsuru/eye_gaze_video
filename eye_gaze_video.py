#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import animatplot as amp
from PIL import Image

#グラフの描画
def plot_animation(ref_df):
    #refの経路描画
    initial = np.array(ref_df["#timestamp"])[0]
    time_data_np = (np.array(ref_df["#timestamp"]) - initial) / 1000
    x_np = np.array(ref_df["gaze_x"])
    y_np = np.array(ref_df["gaze_y"])

    Xs_log = np.array([x_np[t : t + 100] for t in range(len(time_data_np) - 100)])    #X軸データ × 時間軸 分の配列
    Ys_log = np.array([y_np[t : t + 100] for t in range(len(time_data_np) - 100)])    #Y軸データ × 時間軸 分の配列
    #Time_log = np.array([time_data_np[t : t + 100] for t in range(len(time_data_np) - 100)])

    #subplotの描画 (X-Yの情報を3行分の画面で表示)
    fig, ax1 = plt.subplots()
    #ax2 = plt.subplot2grid((3,2), (0,1))
    #ax3 = plt.subplot2grid((3,2), (1,1))
    #ax4 = plt.subplot2grid((3,2), (2,1))

    ax1.set_xlim(0, 1900)    #描画範囲の設定
    ax1.set_ylim(0, 1000)    #描画範囲の設定
    ax1.invert_yaxis()

    im = Image.open("001_back.png")
    ax1.imshow(im)

    block1 = amp.blocks.Scatter(Xs_log, Ys_log, label = "eye_gaze", ax = ax1)
    fig.legend()

    #Time = amp.Timeline(time_data_np[0 : len(time_data_np) - 100], fps = 100)
    anim = amp.Animation([block1])
    anim.controls()
    #anim.save_gif('eye_gaze_video')
    anim.save('eye_gaze_video.mp4')

    plt.show()



if __name__ == '__main__':
    
    csv_file_path = "./tobii_gaze.csv"

    #CSVの読み込み
    ref_df = pd.read_csv(csv_file_path, encoding = "utf-8-sig")    #日本語データ(Shift-Jis)を含む場合を想定
    plot_animation(ref_df)