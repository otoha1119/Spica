# Spica
fix：バグ修正  
hotfix：クリティカルなバグ修正  
add：新規（ファイル）機能追加  
update：機能修正（バグではない）  
change：仕様変更  
clean：整理（リファクタリング等）  
disable：無効化（コメントアウト等）  
remove：削除（ファイル）  
upgrade：バージョンアップ  
revert：変更取り消し  
  
Program実行　python3 -m sdflow.train_sdflow  
Tensofrboard実行　tensorboard --logdir /workspace/checkpoints/logs  
推論実行　python3 -m sdflow.inference  
