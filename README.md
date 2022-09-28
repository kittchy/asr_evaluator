# ESPnet の Evaluation のみを行う

Sagemaker の推論ステップで推論と CER の評価部分だけの実装を行ったので、使いやすいようにドキュメントとしてまとめておきます。

## Inference & scorering

1. ESPnet の python 環境構築

ESPnet の[Install](https://espnet.github.io/espnet/installation.html)をもとに環境を作成してください。以下のコマンドを叩いてみて、arguments が表示されれば OK です。

```
python -c "from espnet2.bin.asr_inference import inference, get_parser"
```

2.  CER の計算の実行環境構築

SCTK をダウンロードします。

```
# SCTK (sclite: CERなどの計算に必要)
  cd <任意のパス>
  git clone https://github.com/usnistgov/SCTK.git \
  && cd SCTK \
  && make config \
  && make all \
  && make check \
  && make install \
  && make doc

  PATH=$PATH:<任意のパス>/SCTK/bin
```

以下のコマンドを使って Argument が出力されれば OK です

```bash
sclite
```

3. スクリプト

このディレクトリにあるコードを実行します。パラメータは以下のとおりです。

```bash
python evaluation.py \
	--asr_train_config /path/to/asr_conf.yml \
	--asr_model_file  /path/to/asr_model.pth \
	--lm_train_config /path/to/lm_conf.yml \
	--lm_train_file /path/to/lm_model.pth \
	--config /path/to/decode_asr.yaml \
	--batch_size 1 \
	--ngpu 0 \
	--data_path_and_name_and_type /path/to/test/wav.scp,speech,sound
	--output_dir /path/to/result
	--reference_dir /path/to/result
	--score_output /path/to/score
```

4. スコアの確認

/path/to/score に結果のファイルがあるので、ご確認ください。
