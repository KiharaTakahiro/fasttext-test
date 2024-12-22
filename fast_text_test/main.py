import re
import MeCab
import ipadic
import os
import fasttext
from typing import List

class TrainDataEntity():
  category: str
  contents: List[str]

  @staticmethod
  def createForDict(map: dict):
    if 'category' not in map:
      raise ApplicationException('学習データのカテゴリが存在しません')
    if 'contents' not in map:
      raise ApplicationException('学習データのコンテンツが存在しません')
    trainDataEntity = TrainDataEntity()
    trainDataEntity.category = map['category']
    trainDataEntity.contents = map['contents']
    return trainDataEntity

  def __str__(self):
    return f'{{category: {self.category} contents: {self.contents}}}'

  def __repr__(self):
    return f'{{category: {self.category} contents: {self.contents}}}'

import datetime
class Logger():

  def __init__(self, log_directory: str = './', log_name: str = 'message.log'):
    self._log_directory = log_directory
    self._log_name = log_name
    self._fullpath = f'{self._log_directory}/{self._log_name}'
  
  def _out_put(self, log_type: str, log_content: str):
    with open(self._fullpath, 'a', encoding='utf-8') as f:
      print(f'[{log_type} {datetime.datetime.now()}] {log_content}', file=f)

  def error(self, log_content: str):
    self._out_put('ERROR', log_content)  
  
  def info(self, log_content: str):
    self._out_put('INFO', log_content)
  
  def clear(self):
    if os.path.isfile(self._fullpath):
      os.remove(self._fullpath)

class ApplicationException(Exception):
  pass

import inspect
class FastTextProcessor():

  # fasttextにて使用するLABELの定数
  LABEL_NAME = "__label__"
  # ディレクトリ
  BASE_DEIRECTORY = "."

  def __init__(self, log_directory: str = BASE_DEIRECTORY, log_name: str = 'message.log'):
    self._logger: Logger = Logger(log_directory, log_name)
    self._tagger = MeCab.Tagger(ipadic.MECAB_ARGS)

  def main_logic(func):
      def wrapper(self, *args, **kwargs):
        arg_names = inspect.getfullargspec(func).args
        arg_values = list(args) + list(kwargs.values())
        arg_dict = dict(zip(arg_names, arg_values))
        try:
          print('処理を開始します')
          self._logger.info(f'{func.__name__} 開始 引数: {arg_dict}')
          rtn = func(self, *args, **kwargs)
          self._logger.info(f'{func.__name__} 終了 引数: {arg_dict}')
          print('処理が終了しました')
          return rtn
        except ApplicationException as e:
          self._logger.error(str(e))
          return f"エラーが発生しました: {str(e)}"
      return wrapper

  def log_clear(self):
    """ 作成したログをクリアする際に使用する

    Raises:
        Exception: _description_
        ApplicationException: _description_
        ApplicationException: _description_
        e: _description_
        ApplicationException: _description_

    Returns:
        _type_: _description_
    """
    self._logger.clear()

  @main_logic
  def predict(self, model_path: str, content: str):
    """識別処理

    Args:
        model_path (str): _description_
        content (str): _description_

    Raises:
        Exception: _description_

    Returns:
        _type_: _description_
    """
    if not os.path.isfile(model_path):
      raise ApplicationException(f"学習データが存在しません model_path: {model_path}")
    model = fasttext.load_model(model_path)
    pre_data = self._cleansing(content)
    estimate = model.predict(pre_data)
    predict = {}
    for (label, degree) in zip(estimate[0], estimate[1]):
      label = re.sub(f"^{self.LABEL_NAME}", '', label)
      predict[label] = degree
    return predict

  def train_for_list(self, trains: list, file_name="test", model_name="test_model"):
    """ listをもとに学習
        学習データの例)
        [{"category": "ラーメンの予約", "contents": ["ラーメンが食べたい", "野菜ラーメンが食べたい", "焼豚麺食べたい"]},
         {"category": "ごはんものの予約", "contents": ["チャーハンが食べたい", "五目御飯が食べたい", "マーボー丼が食べたい"]}]
    Args:
        trains (list): 学習データ
        file_name (str, optional): 学習データの一時ファイル名. Defaults to "test".
        model_name (str, optional): モデルデータ. Defaults to "test_model".
    """
    # 学習処理用にデータ変換
    self._logger.info(f'辞書データをもとに学習可能なデータリストに変換 開始 trains: {trains} file_name: {file_name} model_name: {model_name}')
    new_trains = []
    for train in trains:
      new_trains.append(TrainDataEntity.createForDict(train))
    self._logger.info(f'辞書データをもとに学習可能なデータリストに変換 終了')

    # 学習処理実行
    return self.train(new_trains, file_name, model_name)

  @main_logic
  def train(self, trains: List[TrainDataEntity], file_name="test", model_name="test_model"):
    """ 学習

    Args:
        trains (List[TrainDataEntity]): 学習データ
        file_name (str, optional): 学習データの一時ファイル名. Defaults to "test".
        model_name (str, optional): モデルデータ. Defaults to "test_model".

    Returns:
        _type_: _description_
    """
    # 学習データの一時ファイルが存在している場合は削除する
    path = f"{self.BASE_DEIRECTORY}/{file_name}.txt"
    if os.path.isfile(path):
      os.remove(path)
      self._logger.info(f'前回学習データの一時ファイルを削除しました。削除パス: {path}')

    # 過去のモデルデータが存在している場合は削除する
    model_path= f"{self.BASE_DEIRECTORY}/{model_name}.bin"
    if os.path.isfile(model_path):
      os.remove(model_path)
      self._logger.info(f'前回学習データの一時モデルデータを削除しました。削除パス: {model_path}')

    try:
      # 学習用の事前データ作成
      for train_data in trains:
        self._logger.info(f'学習データ作成 train_data: {train_data}')
        # コンテンツごとに学習データ生成
        for content in train_data.contents:
          # 学習しやすい文言に修正
          pre_data = self._cleansing(content)
          # 対象の文章を分かち書き
          tokens = self._tokenization(pre_data)
          # 学習データに変換
          edit_data = self._edit_train_data(train_data.category, tokens)
          with open(path, 'a', encoding='utf-8') as f:
            print(edit_data, file=f)

      # モデルを学習し保存
      self._logger.info(f'モデルの作成 開始 学習データパス: {path}')
      # この辺のパラメータはなんとなくなので要調整
      model = fasttext.train_supervised(input=path, epoch=20, wordNgrams=2)
      self._logger.info(f'モデルの作成 終了 学習データパス: {path}')
      self._logger.info(f'モデルの保存 開始 モデルデータパス: {model_path}')
      model.save_model(model_path)
      self._logger.info(f'モデルの保存 終了 モデルデータパス: {model_path}')
      return model_path

    except ApplicationException as e:
      raise e
    except Exception as e:
      self._logger.error(str(e))
      raise ApplicationException("想定外のエラーが発生しました") from e
    finally:
      if os.path.isfile(path):
        os.remove(path)
        self._logger.info(f'今回学習データの一時ファイルを削除しました。削除パス: {path}')


  def _edit_train_data(self, category: str, tokens: List[str]) -> str:
    """ fasttextの教師ありデータの形式に変換

    Args:
        category (str): 学習対象のカテゴリ
        tokens (List[str]): 文章を分かち書きした結果

    Returns:
        str: fasttextの教師ありデータの形式変換した値
    """
    self._logger.info(f'fasttext用形式変換 開始 変換前のデータ: {category}')
    edit_data =  f"{self.LABEL_NAME}{category} {' '.join(tokens)}"
    self._logger.info(f'fasttext用形式変換 終了 変換後のデータ: {edit_data}')
    return edit_data

  def _tokenization(self, content: str) -> List[str]:
    """ 文章を分かち書きする

    Args:
        content (str): 分かち書きする対象の文章を指定する

    Returns:
        List[str]: 文章を分かち書きした後の文言を返却する
    """
    self._logger.info(f'分かち書き 開始 分かち書き前のデータ: {content}')
    tokens = []
    node = self._tagger.parseToNode(content)
    previous_node_name = ''
    while node:
      if not '' == node.surface:
        node_names = node.feature.split(",")[0]
        print(f'{previous_node_name} {node.surface} {node_names}')
        # 連続する文字が名詞だったらつなげるいいかどうかは別としてなんとなく
        if previous_node_name == '名詞' and node_names == '名詞':
          tokens[-1] = tokens[-1] + node.surface
        else:
          tokens.append(node.surface)
          previous_node_name = node_names
      node = node.next
    self._logger.info(f'分かち書き 終了 分かち書き後のデータ: {tokens}')
    return tokens

  def _cleansing(self, content: str) -> str:
    """ 学習データのクレンジング

    Args:
        content (str): 文章

    Returns:
        str: 学習用に変換した文章
    """
    self._logger.info(f'学習データのクレンジング 開始 クレンジング前のデータ: {content}')
    # 半角記号の除去
    content = re.sub(r'[!-/:-@[-`{-~]', '', content)
    # 全角記号の除去
    content = re.sub(r'/[！-／：-＠［-｀｛-～、-〜”’・]', '', content)
    self._logger.info(f'学習データのクレンジング 開始 クレンジング後のデータ: {content}')
    return content
