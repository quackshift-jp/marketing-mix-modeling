# マーケティング・ミックス・モデリングアプリケーション
【メイン機能】
・チェネルの売上貢献度を可視化する
・最適なコスト配分を提案
・最適なコスト配分による売上予測を実施

## 使用デモ
使用動画を貼り付けたい.

## 使用技術
Python 3.9.12
Pythonライブラリ（バージョンはrequirements.txtに記載）
フレームワークはStreamlit[https://streamlit.io/]を採用

## ディレクトリ構成
root
app.py
  ┗ features(各機能、もし機能が更に分岐しそうならリファクタリングを実施する)
    ┗ 処理ファイル1.py
    ┗ ...
  ┗ pages(ページ表示を実装)
    ┗ ページファイル.py
    ┗ ...
  ┗ models(データモデル・ビジネスロジックを実装)
    ┗ ...

## 開発用ドキュメント
### Step1: git cloneにて、リポジトリをローカル環境に複製する & 作業ブランチを切る
```
$ git clone https://github.com/quackshift-jp/marketing-mix-modeling.git
$ cd path/to/marketing-mix-modeling && git switch main
$ git pull
$ git checkout -b feature/hogehoge_YYYYMMDD
```

### Step2: Python仮想環境のアクティベート
```
$ python3 -m venv .venv && source .venv/bin/activate
$ pip install -r requirements.txt
```

### Step3: streamlitを起動する
```
$ streamlit run app.py
```