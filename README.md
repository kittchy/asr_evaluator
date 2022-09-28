# ESPnetのEvaluationのみを行う

Sagemakerの推論ステップで推論とCERの評価部分だけの実装を行ったので、使いやすいようにドキュメントとしてまとめておきます。

## Inference

1. ESPnet のpython環境構築

  ESPnetの[Install](https://espnet.github.io/espnet/installation.html)をもとに環境を作成してください。以下のコマンドを叩いてみて、argumentsが表示されればOKです。

```
python -c "from espnet2.bin.asr_inference import inference, get_parser"
```
3.  CERの計算の実行環境構築

  SCTKをダウンロードします。
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
  
  以下のコマンドを使ってArgumentが出力されればOKです

  ```bash
  sclite
  ```
	
2. スクリプト

  このディレクトリにあるコードを実行します。パラメータは以下のとおりです。

  ```bash
  python evaluation.py \
	--asr_train_config \
	--asr_model_file \
	--lm_train_config \
	--lm_train_file \


	--config /opt/ml/processing/input/conf/decode_asr.yaml 
	--batch_size 1 
	--ngpu 0 
	--data_path_and_name_and_type /opt/ml/processing/input/test/wav.scp,speech,sound
	--asr_train_config config.yaml 
	--asr_model_dir /opt/ml/processing/input/asr_model 
	--lm_train_config config.yaml 
	--lm_dir /opt/ml/processing/input/language_model 
	--output_dir /opt/ml/processing/output/test 
	--reference_dir /opt/ml/processing/input/test 
	--score_output /opt/ml/processing/output/score
  ```

# asr_inference
