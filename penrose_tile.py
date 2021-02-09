import math
import tkinter as tk
from collections import namedtuple
from tkinter import colorchooser
from tkinter import filedialog

import cv2
import numpy as np
from PIL import Image, ImageTk

# 点と三角形のための簡易クラス
Point = namedtuple('Point', 'x y')
Triangle = namedtuple('Triangle', 'type p1 p2 p3')

# タイルを構成する三角形の種類 (鋭角または鈍角)
ACUTE = 0
OBTUSE = 1

# 黄金比
GR = (1 + math.sqrt(5)) / 2

# タイルの種類 ("Kite and Dart" または "細いひし形と太いひし形")
K_AND_D = 0
RHOMBUSES = 1


class Pattern(object):
    """タイルを構成する三角形の並びを管理するクラス
    """
    def __init__(self, type, size, colors):
        # タイルの種類
        self.type = type

        # 各タイルの色
        self.set_colors_from_rgb(colors)

        # 最初に作る三角形のオフセットとスケール
        offset = Point(size.x / 2, size.y / 2)
        scale = max(offset.x, offset.y) * 1.5

        # 初期状態として、10個の三角形を作る
        self.triangles = []
        for i in range(10):
            # 配置する角度
            a1 = i * math.pi / 5
            a2 = ((i + 1) % 10) * math.pi / 5

            # 原点を p1 とした時の p2, p3
            p2 = Point(scale * math.cos(a1), scale * math.sin(a1))
            p3 = Point(scale * math.cos(a2), scale * math.sin(a2))
            # 1個おきに反転した三角形とする
            if i % 2 == 0:
                p2, p3 = p3, p2

            # オフセットを適用
            p1 = offset
            p2 = Point(p2.x + offset.x, p2.y + offset.y)
            p3 = Point(p3.x + offset.x, p3.y + offset.y)

            # 三角形を ACUTE タイプとして追加
            type = ACUTE
            self.triangles.append(Triangle(type, p1, p2, p3))

    def set_colors_from_rgb(self, colors):
        """colors に入っている (r, g, b) を逆順にして属性にセットする
        """
        color1 = colors[0][2], colors[0][1], colors[0][0]
        color2 = colors[1][2], colors[1][1], colors[1][0]
        self.colors = (color1, color2)

    def draw(self, img):
        """img 配列に対して現在の状態を描画する
        """
        for tr in self.triangles:
            self.draw_triangle(img, tr)

    def draw_triangle(self, img, tr):
        """三角形を描画する
        """
        # 塗りつぶし
        pts = np.array([tr.p1, tr.p2, tr.p3], np.int32)
        pts = pts.reshape((-1, 1, 2))
        color = self.colors[tr.type]
        cv2.fillConvexPoly(img, pts, color)

        # 輪郭
        if self.type == K_AND_D:
            # Kite and Dart: 各三角形の p1-p2 の辺は描かない
            pts = np.array([tr.p2, tr.p3, tr.p1], np.int32)
        else:
            # 細いひし形と太いひし形: 各三角形の p2-p3 の辺は描かない
            pts = np.array([tr.p3, tr.p1, tr.p2], np.int32)

        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(img, [pts], False, (0, 0, 0), 1)


    def subdivide(self):
        """すべての三角形を分割して、収縮のステップを進める
        """
        if self.type == K_AND_D:
            self.subdivide_k_and_d()
        else:
            self.subdivide_rhombuses()

    def subdivide_k_and_d(self):
        """Kite and Dart の時の三角形の分割
        """
        result = []
        for tr in self.triangles:
            if tr.type == ACUTE:
                # 鋭角の三角形を3つに分割する
                p4_x = tr.p3.x + (tr.p1.x - tr.p3.x) / GR
                p4_y = tr.p3.y + (tr.p1.y - tr.p3.y) / GR
                p4 = Point(p4_x, p4_y)
                p5_x = tr.p1.x + (tr.p2.x - tr.p1.x) / GR
                p5_y = tr.p1.y + (tr.p2.y - tr.p1.y) / GR
                p5 = Point(p5_x, p5_y)
                result.append(Triangle(OBTUSE, p4, tr.p1, p5))
                result.append(Triangle(ACUTE, tr.p3, p5, p4))
                result.append(Triangle(ACUTE, tr.p3, p5, tr.p2))
            else:
                # 鈍角の三角形を2つに分割する
                p4_x = tr.p2.x + (tr.p3.x - tr.p2.x) / GR
                p4_y = tr.p2.y + (tr.p3.y - tr.p2.y) / GR
                p4 = Point(p4_x, p4_y)
                result.append(Triangle(ACUTE, tr.p2, tr.p1, p4))
                result.append(Triangle(OBTUSE, p4, tr.p3, tr.p1))

        self.triangles = result

    def subdivide_rhombuses(self):
        """細いひし形と太いひし形の時の三角形の分割
        """
        result = []
        for tr in self.triangles:
            if tr.type == ACUTE:
                # 鋭角の三角形を2つに分割する
                p4_x = tr.p1.x + (tr.p3.x - tr.p1.x) / GR
                p4_y = tr.p1.y + (tr.p3.y - tr.p1.y) / GR
                p4 = Point(p4_x, p4_y)
                result.append(Triangle(ACUTE, tr.p2, tr.p3, p4))
                result.append(Triangle(OBTUSE, p4, tr.p2, tr.p1))
            else:
                # 鈍角の三角形を3つに分割する
                p4_x = tr.p2.x + (tr.p1.x - tr.p2.x) / GR
                p4_y = tr.p2.y + (tr.p1.y - tr.p2.y) / GR
                p4 = Point(p4_x, p4_y)
                p5_x = tr.p2.x + (tr.p3.x - tr.p2.x) / GR
                p5_y = tr.p2.y + (tr.p3.y - tr.p2.y) / GR
                p5 = Point(p5_x, p5_y)
                result.append(Triangle(ACUTE, p5, tr.p1, p4))
                result.append(Triangle(OBTUSE, p4, p5, tr.p2))
                result.append(Triangle(OBTUSE, p5, tr.p3, tr.p1))

        self.triangles = result


class MainWindow(tk.Tk):
    """ウィンドウ
    """
    def __init__(self):
        super().__init__()
        self.minsize(600, 400)
        self.title("Penrose tile")

        # パターンタイプ (ラジオボタンと連動する値)
        self.ptn_type = tk.IntVar()
        self.ptn_type.set(K_AND_D)

        # 各タイルの色
        self.color1 = ((192, 240, 255), '#c0f0ff')
        self.color2 = ((176, 255, 192), '#b0ffc0')

        # ボタン用のフレーム (サイドバー)
        self.button_frame = tk.Frame()
        self.button_frame.pack(padx=5, pady=5, side=tk.LEFT, fill=tk.Y)

        # キャンバス用のフレーム
        self.canvas_frame = tk.Frame()
        self.canvas_frame.pack(
            padx=5, pady=5, side=tk.LEFT, fill=tk.BOTH, expand=1)

        # ラジオボタン (パターンタイプ)
        self.ptn_rdb_k_and_d = tk.Radiobutton(
            self.button_frame,
            value=K_AND_D, variable=self.ptn_type, text='Kite and Dart')
        self.ptn_rdb_k_and_d.pack(padx=5, pady=5, side=tk.TOP)

        self.ptn_rdb_rhombuses = tk.Radiobutton(
            self.button_frame,
            value=RHOMBUSES, variable=self.ptn_type, text='Rhombuses')
        self.ptn_rdb_rhombuses.pack(padx=5, pady=5, side=tk.TOP)

        # 初期化ボタン
        self.ini_btn = tk.Button(
            self.button_frame,
            text='Initialize', command=self.initialize, width=10)
        self.ini_btn.pack(padx=5, pady=5, side=tk.TOP)

        # 収縮ボタン
        self.def_btn = tk.Button(
            self.button_frame,
            text='Deflate', command=self.deflate, width=10)
        self.def_btn.pack(padx=5, pady=5, side=tk.TOP)

        # カラーピッカー1
        self.col_btn_1 = tk.Button(
            self.button_frame,
            text='Color 1', command=self.pick_col_1, width=10,
            bg=self.color1[1])
        self.col_btn_1.pack(padx=5, pady=5, side=tk.TOP)

        # カラーピッカー2
        self.col_btn_2 = tk.Button(
            self.button_frame,
            text='Color 2', command=self.pick_col_2, width=10,
            bg=self.color2[1])
        self.col_btn_2.pack(padx=5, pady=5, side=tk.TOP)

        # 再描画ボタン
        self.redaw_btn = tk.Button(
            self.button_frame,
            text='Redraw', command=self.redraw, width=10)
        self.redaw_btn.pack(padx=5, pady=5, side=tk.TOP)

        # 保存ボタン
        self.save_btn = tk.Button(
            self.button_frame,
            text='Save', command=self.save, width=10, state=tk.DISABLED)
        self.save_btn.pack(padx=5, pady=5, side=tk.TOP)

        # キャンバス
        self.canvas = tk.Canvas(self.canvas_frame, bg='white')
        self.canvas.pack(padx=5, pady=5, fill=tk.BOTH, expand=1)

    def initialize(self):
        """初期状態のパターンを作ってキャンバスを初期化する
        """
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        type = self.ptn_type.get()
        colors = self.color1[0], self.color2[0]

        # 初期状態のパターンを生成する
        self.pattern = Pattern(type, Point(w, h), colors)

        # キャンバスに描画する
        self.draw_pattern()

    def deflate(self):
        """収縮 (三角形を分割) する
        """
        self.pattern.subdivide()
        self.draw_pattern()

    def pick_col_1(self):
        """カラーピッカーを開いて、選択された値を Color1 にセットする
        """
        color_code = colorchooser.askcolor(self.color1[0])
        if all(color_code):
            rgb = [int(x) for x in color_code[0]]
            self.color1 = (tuple(rgb), color_code[1])
            self.col_btn_1['bg'] = color_code[1]

    def pick_col_2(self):
        """カラーピッカーを開いて、選択された値を Color2 にセットする
        """
        color_code = colorchooser.askcolor(self.color2[0])
        if all(color_code):
            rgb = [int(x) for x in color_code[0]]
            self.color2 = (tuple(rgb), color_code[1])
            self.col_btn_2['bg'] = color_code[1]

    def draw_pattern(self):
        """現在のパターンをキャンバスに描画する
        """
        # キャンバスのサイズに合わせて白いオフスクリーンを作る
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        img = np.full((h, w, 3), 255, np.uint8)

        # オフスクリーンに描画する
        self.pattern.draw(img)

        # オフスクリーンを変形してキャンバスに置く
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.image_pil = Image.fromarray(image_rgb)
        self.image_tk = ImageTk.PhotoImage(self.image_pil)
        self.canvas.create_image(0, 0, image=self.image_tk, anchor=tk.NW)

        # 保存ボタンを有効にする
        self.save_btn['state'] = tk.NORMAL

    def redraw(self):
        """選択中の色を使って現在のパターンを再描画する
        """
        if not hasattr(self, 'pattern'):
            return
        colors = self.color1[0], self.color2[0]
        self.pattern.set_colors_from_rgb(colors)
        self.draw_pattern()

    def save(self):
        """表示中の画像をファイルに保存する
        """
        if not hasattr(self, 'image_pil'):
            return
        filename = filedialog.asksaveasfilename(
            initialdir=".",
            title="Save as",
            filetypes=[
                ("PNG Image Files", ".png"),
                ("JPEG Image Files", ".jpg"),
                ("GIF Image Files", ".gif"),
            ],
            initialfile='New_Image',
            defaultextension = ".png"
        )
        if filename:
            self.image_pil.save(filename)


if __name__ == '__main__':
    root = MainWindow()
    root.mainloop()
