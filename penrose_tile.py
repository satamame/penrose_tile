import math
import tkinter as tk
from collections import namedtuple
from tkinter import colorchooser
from tkinter import filedialog

import cv2
import numpy as np
from PIL import Image, ImageTk

# 点, サイズ, 三角形のための簡易クラス
Point = namedtuple('Point', 'x y')
Size = namedtuple('Size', 'w, h')
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
    def __init__(self, type_, size, colors, ltype):
        # タイルの種類
        self.type = type_

        # 各タイルの色
        self.set_colors_from_rgb(colors)

        # ラインタイプ
        self.set_ltype(ltype)

        # 最初に作る三角形のオフセットとスケール
        offset = Point(size.w / 2, size.h / 2)
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
            self.triangles.append(Triangle(ACUTE, p1, p2, p3))

    def set_colors_from_rgb(self, colors):
        """colors に入っている (r, g, b) を逆順にして属性にセットする
        """
        self.colors = [list(reversed(c)) for c in colors]

    def set_ltype(self, ltype):
        """ラインタイプを属性にセットする

        ラインタイプは ('LINE_4', 'LINE_8', 'LINE_AA') のいずれか
        """
        self.ltype = getattr(cv2, ltype)

    def draw(self, img):
        """img 配列に対して現在の状態を描画する
        """
        # 輪郭はリストにまとめて最後に一括で描画する
        polylines = []

        # 描くべき辺を取り出す関数 (タイルの種類によって違う)
        if self.type == K_AND_D:
            def get_line(tr):
                return np.array([tr.p2, tr.p3, tr.p1], np.int32)
        else:
            def get_line(tr):
                return np.array([tr.p3, tr.p1, tr.p2], np.int32)

        # すべての三角形を描画するループ
        for tr in self.triangles:
            # 塗りつぶし
            pts = np.array([tr.p1, tr.p2, tr.p3], np.int32)
            pts = pts.reshape((-1, 1, 2))
            color = self.colors[tr.type]
            cv2.fillConvexPoly(img, pts, color)

            # 描くべき辺を取り出してリストに追加
            polyline = get_line(tr).reshape((-1, 1, 2))
            polylines.append(polyline)

        # 輪郭を描画する
        cv2.polylines(img, polylines, False, self.colors[2], 1, self.ltype)

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


# ウィンドウのモード
MODE_SETTING = 0  # 設定中
MODE_DRAWING = 1  # 描画中


def validate_digit(before, after):
    """数字入力のバリデーション
    """
    return after.isdigit() or not after


class MainWindow(tk.Tk):
    """ウィンドウ
    """
    def __init__(self):
        super().__init__()
        self.minsize(593, 397)
        self.title("Penrose tile")

        # モード
        self.mode = MODE_SETTING

        # パターンタイプ (ラジオボタンと連動する値)
        self.ptn_type = tk.IntVar(value=K_AND_D)

        # 画像サイズをキャンバスに合わせる (チェックボックスと連動する値)
        self.fit_canvas = tk.BooleanVar(value=True)

        # 色 (タイル1, タイル2, ライン)
        self.colors = [
            ((192, 240, 255), '#c0f0ff'),
            ((176, 255, 192), '#b0ffc0'),
            ((0, 0, 0), '#000000'),
        ]

        # ラインタイプ (プルダウンに設定する選択肢)
        self.ltypes = (
            'LINE_4',
            'LINE_8',
            'LINE_AA',
        )

        # ラインタイプ (プルダウンと連動する値)
        self.ltype = tk.StringVar(value=self.ltypes[1])

        # GUI の初期化
        self.init_gui()

    def init_gui(self):
        """GUI の初期化
        """
        # ボタン用のフレーム (サイドバー)
        self.button_frame = tk.Frame()
        self.button_frame.pack(padx=5, pady=5, side=tk.LEFT, fill=tk.Y)

        # キャンバス用のフレーム
        self.canvas_frame = tk.Frame()
        self.canvas_frame.pack(
            padx=10, pady=10, side=tk.LEFT, fill=tk.BOTH, expand=1)

        # リセットボタン
        self.reset_btn = tk.Button(
            self.button_frame,
            text='Reset', command=self.reset, width=10, state=tk.DISABLED)
        self.reset_btn.pack(padx=5, pady=5, side=tk.TOP)

        # ラジオボタン (パターンタイプ)
        self.ptn_rdb_k_and_d = tk.Radiobutton(
            self.button_frame,
            value=K_AND_D, variable=self.ptn_type, text='Kite and Dart')
        self.ptn_rdb_k_and_d.pack(padx=5, pady=0, side=tk.TOP, anchor=tk.W)

        self.ptn_rdb_rhombuses = tk.Radiobutton(
            self.button_frame,
            value=RHOMBUSES, variable=self.ptn_type, text='Rhombuses')
        self.ptn_rdb_rhombuses.pack(padx=5, pady=0, side=tk.TOP, anchor=tk.W)

        # チェックボックス (画像サイズをキャンバスに合わせる)
        self.fit_chk = tk.Checkbutton(
            self.button_frame, command=self.check_fit,
            variable=self.fit_canvas, text='Canvas size')
        self.fit_chk.pack(padx=5, pady=0, side=tk.TOP, anchor=tk.W)

        # サイズ指定 UI 群フレーム
        dims_frame = tk.Frame(self.button_frame)
        dims_frame.pack(padx=5, pady=5, side=tk.TOP)

        # ラベル (w:)
        w_lbl = tk.Label(dims_frame, text='w:')
        w_lbl.pack(side=tk.LEFT, anchor=tk.W)

        # サイズ (w) のテキストフィールド
        self.w_txt = tk.Entry(dims_frame, width=4, state=tk.DISABLED)
        self.w_txt.pack(side=tk.LEFT, anchor=tk.W)
        vcmd_w = (self.w_txt.register(validate_digit), '%s', '%P')
        self.w_txt.configure(validate='key', vcmd=vcmd_w)

        # ラベル (h:)
        h_lbl = tk.Label(dims_frame, text='h:')
        h_lbl.pack(side=tk.LEFT, anchor=tk.W)

        # サイズ (h) のテキストフィールド
        self.h_txt = tk.Entry(dims_frame, width=4, state=tk.DISABLED)
        self.h_txt.pack(side=tk.LEFT, anchor=tk.W)
        vcmd_h = (self.h_txt.register(validate_digit), '%s', '%P')
        self.h_txt.configure(validate='key', vcmd=vcmd_h)

        # 収縮ボタン (最初は初期化ボタン)
        self.def_btn = tk.Button(
            self.button_frame,
            text='Initialize', command=self.deflate, width=10)
        self.def_btn.pack(padx=5, pady=5, side=tk.TOP)

        # カラーピッカー0 (タイル1の色)
        self.col_btn_0 = tk.Button(
            self.button_frame,
            text='Color 0', command=lambda: self.pick_color(0), width=10,
            bg=self.colors[0][1])
        self.col_btn_0.pack(padx=5, pady=5, side=tk.TOP)

        # カラーピッカー1 (タイル2の色)
        self.col_btn_1 = tk.Button(
            self.button_frame,
            text='Color 1', command=lambda: self.pick_color(1), width=10,
            bg=self.colors[1][1])
        self.col_btn_1.pack(padx=5, pady=5, side=tk.TOP)

        # カラーピッカー2 (ラインの色)
        self.col_btn_2 = tk.Button(
            self.button_frame,
            text='Line Color', command=lambda: self.pick_color(2), width=10,
            bg=self.colors[2][1], fg='white')
        self.col_btn_2.pack(padx=5, pady=5, side=tk.TOP)

        # ラインタイプメニュー
        self.ltype_menu = tk.OptionMenu(
            self.button_frame, self.ltype, *self.ltypes)
        self.ltype_menu.pack(padx=5, pady=5, side=tk.TOP)
        self.ltype.trace('w', self.apply_ltype)

        # 保存ボタン
        self.save_btn = tk.Button(
            self.button_frame,
            text='Save', command=self.save, width=10, state=tk.DISABLED)
        self.save_btn.pack(padx=5, pady=5, side=tk.TOP)

        # 縦スクロールバー
        self.bar_y = tk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL)
        self.bar_y.pack(side=tk.RIGHT, fill=tk.Y)

        # 横スクロールバー
        self.bar_x = tk.Scrollbar(self.canvas_frame, orient=tk.HORIZONTAL)
        self.bar_x.pack(side=tk.BOTTOM, fill=tk.X)

        # キャンバス
        self.canvas = tk.Canvas(
            self.canvas_frame, bd=0, highlightthickness=0, bg='white')
        self.canvas.pack(fill=tk.BOTH, expand=1)
        self.canvas.bind('<Configure>', self.canvas_resized)

        # スクロールバーの関連付け
        self.bar_y.config(command=self.canvas.yview)
        self.bar_x.config(command=self.canvas.xview)
        self.canvas.config(
            yscrollcommand=self.bar_y.set, xscrollcommand=self.bar_x.set)

    def reset(self):
        """状態をリセットして設定を編集可能にする
        """
        # 保存ボタンをクリック不可にする
        self.save_btn['state'] = tk.DISABLED

        # モードを「設定中」にする
        self.mode = MODE_SETTING
        self.reset_btn['state'] = tk.DISABLED

        # 描画用の属性をクリアする
        self.pattern = None
        self.image_pil = None
        self.image_tk = None

        # 設定用の UI を編集可能にする
        self.ptn_rdb_k_and_d['state'] = tk.NORMAL
        self.ptn_rdb_rhombuses['state'] = tk.NORMAL
        self.fit_chk['state'] = tk.NORMAL

        self.w_txt['state'] = tk.NORMAL
        self.h_txt['state'] = tk.NORMAL

        # 画像サイズをキャンバスに合わせているなら、UI を更新して編集不可にする
        if self.fit_canvas.get():
            self.indicate_canvas_size()
            self.w_txt['state'] = tk.DISABLED
            self.h_txt['state'] = tk.DISABLED

        # キャンバスのスクロール範囲をリセット
        self.canvas.config(scrollregion=(0, 0, 0, 0))

        # 収縮ボタンのテキストを変更する
        self.def_btn['text'] = 'Initialize'

    def canvas_resized(self, event):
        """キャンバスサイズが変わった時のイベントハンドラ
        """
        # 設定変更中で、画像サイズをキャンバスに合わせているなら、UI を更新
        if self.mode == MODE_SETTING and self.fit_canvas.get():
            self.w_txt['state'] = tk.NORMAL
            self.h_txt['state'] = tk.NORMAL
            self.indicate_canvas_size()
            self.w_txt['state'] = tk.DISABLED
            self.h_txt['state'] = tk.DISABLED

    def check_fit(self):
        """チェックボックス (画像サイズをキャンバスに…) のクリックハンドラ
        """
        # チェックしたなら、キャンバスサイズを UI に反映して編集不可にする
        if self.fit_canvas.get():
            self.indicate_canvas_size()
            self.w_txt['state'] = tk.DISABLED
            self.h_txt['state'] = tk.DISABLED

        # チェックを外したなら、UI を編集可能にする
        else:
            self.w_txt['state'] = tk.NORMAL
            self.h_txt['state'] = tk.NORMAL

    def indicate_canvas_size(self):
        """キャンバスサイズを UI に反映する
        """
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        self.w_txt.delete(0, tk.END)
        self.w_txt.insert(tk.END, str(w))
        self.h_txt.delete(0, tk.END)
        self.h_txt.insert(tk.END, str(h))

    def deflate(self):
        """収縮 (三角形を分割) して描画する
        """
        # モードが設定変更中だった場合の処理
        if self.mode == MODE_SETTING:
            # モードを描画中にする
            self.mode = MODE_DRAWING
            self.reset_btn['state'] = tk.NORMAL

            # w, h が空なら現在のキャンバスのサイズから取る
            if not self.w_txt.get():
                self.w_txt.insert(tk.END, str(self.canvas.winfo_width()))
            if not self.h_txt.get():
                self.h_txt.insert(tk.END, str(self.canvas.winfo_height()))

            # 設定用の UI を編集不可にする
            self.ptn_rdb_k_and_d['state'] = tk.DISABLED
            self.ptn_rdb_rhombuses['state'] = tk.DISABLED
            self.fit_chk['state'] = tk.DISABLED
            self.w_txt['state'] = tk.DISABLED
            self.h_txt['state'] = tk.DISABLED

            # パターンを初期化する
            self.initialize()

            # キャンバスのスクロール範囲を決める
            self.canvas.config(scrollregion=(0, 0, *self.img_size))

            # 保存ボタンをクリック可能にする
            self.save_btn['state'] = tk.NORMAL

            # 収縮ボタンのテキストを変更する
            self.def_btn['text'] = 'Deflate'

        # 収縮と描画の処理
        if getattr(self, 'pattern', None):
            self.pattern.subdivide()
            self.draw_pattern()

    def initialize(self):
        """初期状態のパターンを生成する
        """
        # 画像サイズを確定する
        if self.fit_canvas.get():
            w = self.canvas.winfo_width()
            h = self.canvas.winfo_height()
        else:
            w = int(self.w_txt.get())
            h = int(self.h_txt.get())
        self.img_size = Size(w, h)

        # パラメタを確定する
        type_ = self.ptn_type.get()
        colors = [c[0] for c in self.colors]
        ltype = self.ltype.get()

        # 初期パターンを生成する
        self.pattern = Pattern(type_, self.img_size, colors, ltype)

    def draw_pattern(self):
        """現在のパターンをキャンバスに描画する
        """
        # 白いオフスクリーンを作る
        w, h = self.img_size
        img = np.full((h, w, 3), 255, np.uint8)

        # オフスクリーンに描画する
        self.pattern.draw(img)

        # オフスクリーンを変形してキャンバスに置く
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.image_pil = Image.fromarray(image_rgb)
        self.image_tk = ImageTk.PhotoImage(self.image_pil)
        self.canvas.create_image(0, 0, image=self.image_tk, anchor=tk.NW)

    def pick_color(self, i):
        """カラーピッカーを開いて、色をセットして、再描画する
        """
        color_code = colorchooser.askcolor(self.colors[i][0])
        if all(color_code):
            rgb = [int(x) for x in color_code[0]]
            self.colors[i] = (tuple(rgb), color_code[1])
            self.apply_colors()
            self.update_col_btn(i)

    def apply_colors(self):
        """選択中の色を使って現在のパターンを再描画する
        """
        if getattr(self, 'pattern', None):
            colors = [c[0] for c in self.colors]
            self.pattern.set_colors_from_rgb(colors)
            self.draw_pattern()

    def update_col_btn(self, i):
        """カラーピッカーボタンの色を更新
        """
        # 対象のボタンと色
        btn = getattr(self, f'col_btn_{i}')
        col = self.colors[i]

        # ボタンの背景色をセット
        btn['bg'] = col[1]
        # 文字色を背景の明るさに合わせてセット
        brightness = col[0][0] / 4 + col[0][1] * 1.8 + col[0][2] / 6
        btn['fg'] = 'black' if brightness > 384 else 'white'

    def apply_ltype(self, *args):
        """選択中のラインタイプを使って現在のパターンを再描画する
        """
        if getattr(self, 'pattern', None):
            self.pattern.set_ltype(self.ltype.get())
            self.draw_pattern()

    def save(self):
        """表示中の画像をファイルに保存する
        """
        if not getattr(self, 'image_pil', None):
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
            defaultextension=".png"
        )
        if filename:
            self.image_pil.save(filename)


if __name__ == '__main__':
    root = MainWindow()
    root.mainloop()
