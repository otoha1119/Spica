・Docker関連の使い方
GPUを積んだ学校の端末と自宅のPCの両方で編集できるようにファイルを構成した
また，データセットのPathを.envファイルで切り替えしなければいけない
使用しない方を#でコメントしておくこと

1. Mac用
docker-compose.yaml
requirements.txt

2. Windows用
docker-compose.gpu.yaml
requirements-gpu.txt

・実行コマンド
Mac用
docker-compose -f docker/docker-compose.yaml up -d

Windows用
docker-compose -f docker/docker-compose.gpu.yaml up -d


※dockerfile関連の変更後は
docker-compose -f docker/docker-compose.yaml up --build -d
docker-compose -f docker/docker-compose.gpu.yaml up --build -d

Tensorboard関連

TensorBoard を起動
コンテナ内で以下のコマンドを実行します。
tensorboard --logdir=runs --host=0.0.0.0 --port=6006

ブラウザで確認
ホストマシンのブラウザで以下の URL にアクセスします。
http://<コンテナのIPアドレス>:6006

コンテナの IP アドレスは以下のコマンドで確認できます。
docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' Noctua


