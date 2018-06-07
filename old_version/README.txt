# New
將原始資料進行預處理
1. 只保留中文字
2. 移除標點符號
3. 資料格式[類別, 謎面, 謎底]
4. 輸出成pkl檔
程式 01_clean_data.py
參數
--data_path，資料表位置和資料表，預設".\data\newProverb2_v2_0.csv"
--output_path，為輸出資料的位置
--dict_path，Jieba斷詞所需的詞庫，預設使用Jieba預設詞庫，額外詞庫存放於".\tools\extra_dict\dict.txt"

python 0_clean_data.py --data_path=./data/newProverb2_v2_0.csv --output_path=./pkl_data/newProverb2_jieba_v2_1.pkl --dict_path=./tools/extra_dict/dict.txt

訓練字詞向量
程式 13_train_word2vec.py
--dim_size，預設=100，訓練出的詞向量會有維度數
-sg，預設=1，sg=1表示採用skip-gram,sg=0 表示採用cbow
--workers，預設=4，為執行緒數目
--min_count，預設=5，若這個詞出現的次數小於min_count，那他就不會被視為訓練對象
--window，預設=5，分析字詞的窗口大小
--sentences_path，為分析文本資料的路徑位置
--output_path，為輸出資料的位置

python 13_train_word2vec.py --dim_size=100 -sg=1 --workers=4 --min_count=1 --window=5 --sentences_path=./word2vec_data/segmentation.txt --output_path=./word2vec
python 13_train_word2vec.py --dim_size=300 -sg=1 --workers=4 --min_count=1 --window=5 --sentences_path=segmentation2.txt --output_path=.

# Old
Dataset PreProcessing
執行資料集切割成五等份
程式 datasetCV.py
參數
--analysis_type，切割分析的類型有All、A、B三種，All為謎面和謎底皆有，A為謎面，B為謎底
--data_path，資料表位置和資料表，預設"./Proverb_nonexpansion_all_seq2seq_v1.csv"
--output_path，為輸出資料的位置
--num_CV，交叉驗證的等分數量，預設為5

將訓練和測試的文字資料轉換為字詞向量(以訓練資料集為依據，將測試集轉換為等長)
程式 convert_wordvector.py
參數
--analysis_type，分析類型，與上述參數相同意思。
--data_path，交叉驗證資料的位置
--output_path，為輸出資料的位置
--num_CV，交叉驗證的數量，預設為5
--remove_mid，是否移除中性類別，預設False
--equallabel，以最小類別的數量為依據，預設False

python convert_wordvector.py --data_path=./res --output_path=./res --analysis_type=All --n=None --train_filename=0m_TR --test_filename=0m_TE

A. 無移除中性類別且無均勻分布類別筆數
將完整資料做伍等分交叉驗證
python datasetCV.py --data_path=./Proverb_nonexpansion_all_seq2seq_v1.csv --output_path=./datas --remove_mid=False --equallabel=False --num_CV=5
1.  謎面和謎底版
python convert_wordvector.py --data_path=./datas --output_path=./data --analysis_type=All --num_CV=5
2.  謎面版
python convert_wordvector.py --data_path=./datas --output_path=./dataA --analysis_type=A --num_CV=5
3.  謎底版
python convert_wordvector.py --data_path=./datas --output_path=./dataB --analysis_type=B --num_CV=5

B. 無移除中性類別且有均勻分布類別筆數
python datasetCV.py --data_path=./Proverb_nonexpansion_all_seq2seq_v1.csv --output_path=./datasb --remove_mid=False --equallabel=True --num_CV=5

python convert_wordvector.py --data_path=./datasb --output_path=./databa --analysis_type=All --num_CV=5

python convert_wordvector.py --data_path=./datasb --output_path=./databaA --analysis_type=A --num_CV=5

python convert_wordvector.py --data_path=./datasb --output_path=./databaB --analysis_type=B --num_CV=5

C. 有移除中性類別且無均勻分布類別筆數
python datasetCV.py --data_path=./Proverb_nonexpansion_all_seq2seq_v1.csv --output_path=./datasm --remove_mid=True --equallabel=False --num_CV=5

python convert_wordvector.py --data_path=./datasm --output_path=./datam --analysis_type=All --num_CV=5

python convert_wordvector.py --data_path=./datasm --output_path=./datamA --analysis_type=A --num_CV=5

python convert_wordvector.py --data_path=./datasm --output_path=./datamB --analysis_type=B --num_CV=5

D. 有移除中性類別且有均勻分布類別筆數
python datasetCV.py --data_path=./Proverb_nonexpansion_all_seq2seq_v1.csv --output_path=./datasbm --remove_mid=True --equallabel=True --num_CV=5

python convert_wordvector.py --data_path=./datasbm --output_path=./databam --analysis_type=All --num_CV=5

python convert_wordvector.py --data_path=./datasbm --output_path=./databamA --analysis_type=A --num_CV=5

python convert_wordvector.py --data_path=./datasbm --output_path=./databamB --analysis_type=B --num_CV=5

----------------------------------------------------------------------------------------------------------------------------------------------
RUN Classifier
分類模型
程式 k_test_model5.py
參數
--data_path，為資料表的位置，最後請勿加上斜線，範例：./data/nonbalance/data
--Train_set，訓練資料集的檔名和附檔名，0m_TR.txt
--Test_set，測試資料集的檔名和附檔名0m_TE.txt
--remove_mid，是否移除中性類別，預設False
--epochs_step，跌代次數，預設20
--_batch_size，批次的數量，預設128
--limit_v，針對字詞向量的頻率，設定1為1以上的皆保留。

A. 無移除中性類別且無均勻分布類別筆數
a)謎面和謎底版
python k_test_model6.py --data_path=./data/nonbalance/data --Train_set=m_TR.txt --Test_set=m_TE.txt --remove_mid=False --epochs_step=20 --_batch_size=128 --limit_v=1 --num_CV=5

python k_test_model5.py --data_path=./data/nonbalance/data --Train_set=0m_TR.txt --Test_set=0m_TE.txt --remove_mid=False --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/nonbalance/data --Train_set=1m_TR.txt --Test_set=1m_TE.txt --remove_mid=False --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/nonbalance/data --Train_set=2m_TR.txt --Test_set=2m_TE.txt --remove_mid=False --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/nonbalance/data --Train_set=3m_TR.txt --Test_set=3m_TE.txt --remove_mid=False --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/nonbalance/data --Train_set=4m_TR.txt --Test_set=4m_TE.txt --remove_mid=False --epochs_step=20 --_batch_size=128 --limit_v=1

b)謎面版
python k_test_model6.py --data_path=./data/nonbalance/dataA --Train_set=m_TR.txt --Test_set=m_TE.txt --remove_mid=False --epochs_step=20 --_batch_size=128 --limit_v=1 --num_CV=5

python k_test_model5.py --data_path=./data/nonbalance/dataA --Train_set=0m_TR.txt --Test_set=0m_TE.txt --remove_mid=False --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/nonbalance/dataA --Train_set=1m_TR.txt --Test_set=1m_TE.txt --remove_mid=False --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/nonbalance/dataA --Train_set=2m_TR.txt --Test_set=2m_TE.txt --remove_mid=False --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/nonbalance/dataA --Train_set=3m_TR.txt --Test_set=3m_TE.txt --remove_mid=False --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/nonbalance/dataA --Train_set=4m_TR.txt --Test_set=4m_TE.txt --remove_mid=False --epochs_step=20 --_batch_size=128 --limit_v=1

c)謎底版
python k_test_model6.py --data_path=./data/nonbalance/dataB --Train_set=m_TR.txt --Test_set=m_TE.txt --remove_mid=False --epochs_step=20 --_batch_size=128 --limit_v=1 --num_CV=5

python k_test_model5.py --data_path=./data/nonbalance/dataB --Train_set=0m_TR.txt --Test_set=0m_TE.txt --remove_mid=False --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/nonbalance/dataB --Train_set=1m_TR.txt --Test_set=1m_TE.txt --remove_mid=False --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/nonbalance/dataB --Train_set=2m_TR.txt --Test_set=2m_TE.txt --remove_mid=False --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/nonbalance/dataB --Train_set=3m_TR.txt --Test_set=3m_TE.txt --remove_mid=False --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/nonbalance/dataB --Train_set=4m_TR.txt --Test_set=4m_TE.txt --remove_mid=False --epochs_step=20 --_batch_size=128 --limit_v=1

B. 無移除中性類別且有均勻分布類別筆數
a)謎面和謎底版
python k_test_model6.py --data_path=./data/balance/databa --Train_set=m_TR.txt --Test_set=m_TE.txt --remove_mid=False --epochs_step=20 --_batch_size=128 --limit_v=1 --num_CV=5

python k_test_model5.py --data_path=./data/balance/databa --Train_set=0m_TR.txt --Test_set=0m_TE.txt --remove_mid=False --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/balance/databa --Train_set=1m_TR.txt --Test_set=1m_TE.txt --remove_mid=False --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/balance/databa --Train_set=2m_TR.txt --Test_set=2m_TE.txt --remove_mid=False --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/balance/databa --Train_set=3m_TR.txt --Test_set=3m_TE.txt --remove_mid=False --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/balance/databa --Train_set=4m_TR.txt --Test_set=4m_TE.txt --remove_mid=False --epochs_step=20 --_batch_size=128 --limit_v=1

b)謎面版
python k_test_model6.py --data_path=./data/balance/databaA --Train_set=m_TR.txt --Test_set=m_TE.txt --remove_mid=False --epochs_step=20 --_batch_size=128 --limit_v=1 --num_CV=5

python k_test_model5.py --data_path=./data/balance/databaA --Train_set=0m_TR.txt --Test_set=0m_TE.txt --remove_mid=False --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/balance/databaA --Train_set=1m_TR.txt --Test_set=1m_TE.txt --remove_mid=False --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/balance/databaA --Train_set=2m_TR.txt --Test_set=2m_TE.txt --remove_mid=False --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/balance/databaA --Train_set=3m_TR.txt --Test_set=3m_TE.txt --remove_mid=False --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/balance/databaA --Train_set=4m_TR.txt --Test_set=4m_TE.txt --remove_mid=False --epochs_step=20 --_batch_size=128 --limit_v=1

c)謎底版
python k_test_model6.py --data_path=./data/balance/databaB --Train_set=m_TR.txt --Test_set=m_TE.txt --remove_mid=False --epochs_step=20 --_batch_size=128 --limit_v=1 --num_CV=5

python k_test_model5.py --data_path=./data/balance/databaB --Train_set=0m_TR.txt --Test_set=0m_TE.txt --remove_mid=False --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/balance/databaB --Train_set=1m_TR.txt --Test_set=1m_TE.txt --remove_mid=False --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/balance/databaB --Train_set=2m_TR.txt --Test_set=2m_TE.txt --remove_mid=False --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/balance/databaB --Train_set=3m_TR.txt --Test_set=3m_TE.txt --remove_mid=False --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/balance/databaB --Train_set=4m_TR.txt --Test_set=4m_TE.txt --remove_mid=False --epochs_step=20 --_batch_size=128 --limit_v=1

C. 有移除中性類別且無均勻分布類別筆數
a)謎面和謎底版
python k_test_model6.py --data_path=./data/nonbalance/datam --Train_set=m_TR.txt --Test_set=m_TE.txt --remove_mid=True --epochs_step=20 --_batch_size=128 --limit_v=1 --num_CV=5

python k_test_model5.py --data_path=./data/nonbalance/datam --Train_set=0m_TR.txt --Test_set=0m_TE.txt --remove_mid=True --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/nonbalance/datam --Train_set=1m_TR.txt --Test_set=1m_TE.txt --remove_mid=True --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/nonbalance/datam --Train_set=2m_TR.txt --Test_set=2m_TE.txt --remove_mid=True --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/nonbalance/datam --Train_set=3m_TR.txt --Test_set=3m_TE.txt --remove_mid=True --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/nonbalance/datam --Train_set=4m_TR.txt --Test_set=4m_TE.txt --remove_mid=True --epochs_step=20 --_batch_size=128 --limit_v=1

b)謎面版
python k_test_model6.py --data_path=./data/nonbalance/datamA --Train_set=m_TR.txt --Test_set=m_TE.txt --remove_mid=True --epochs_step=20 --_batch_size=128 --limit_v=1 --num_CV=5

python k_test_model5.py --data_path=./data/nonbalance/datamA --Train_set=0m_TR.txt --Test_set=0m_TE.txt --remove_mid=True --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/nonbalance/datamA --Train_set=1m_TR.txt --Test_set=1m_TE.txt --remove_mid=True --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/nonbalance/datamA --Train_set=2m_TR.txt --Test_set=2m_TE.txt --remove_mid=True --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/nonbalance/datamA --Train_set=3m_TR.txt --Test_set=3m_TE.txt --remove_mid=True --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/nonbalance/datamA --Train_set=4m_TR.txt --Test_set=4m_TE.txt --remove_mid=True --epochs_step=20 --_batch_size=128 --limit_v=1

c)謎底版
python k_test_model6.py --data_path=./data/nonbalance/datamB --Train_set=m_TR.txt --Test_set=m_TE.txt --remove_mid=True --epochs_step=20 --_batch_size=128 --limit_v=1 --num_CV=5

python k_test_model5.py --data_path=./data/nonbalance/datamB --Train_set=0m_TR.txt --Test_set=0m_TE.txt --remove_mid=True --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/nonbalance/datamB --Train_set=1m_TR.txt --Test_set=1m_TE.txt --remove_mid=True --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/nonbalance/datamB --Train_set=2m_TR.txt --Test_set=2m_TE.txt --remove_mid=True --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/nonbalance/datamB --Train_set=3m_TR.txt --Test_set=3m_TE.txt --remove_mid=True --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/nonbalance/datamB --Train_set=4m_TR.txt --Test_set=4m_TE.txt --remove_mid=True --epochs_step=20 --_batch_size=128 --limit_v=1

D. 有移除中性類別且有均勻分布類別筆數
a)謎面和謎底版
python k_test_model6.py --data_path=./data/balance/databam --Train_set=m_TR.txt --Test_set=m_TE.txt --remove_mid=True --epochs_step=20 --_batch_size=128 --limit_v=1 --num_CV=5

python k_test_model5.py --data_path=./data/balance/databam --Train_set=0m_TR.txt --Test_set=0m_TE.txt --remove_mid=True --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/balance/databam --Train_set=1m_TR.txt --Test_set=1m_TE.txt --remove_mid=True --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/balance/databam --Train_set=2m_TR.txt --Test_set=2m_TE.txt --remove_mid=True --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/balance/databam --Train_set=3m_TR.txt --Test_set=3m_TE.txt --remove_mid=True --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/balance/databam --Train_set=4m_TR.txt --Test_set=4m_TE.txt --remove_mid=True --epochs_step=20 --_batch_size=128 --limit_v=1

b)謎面版
python k_test_model6.py --data_path=./data/balance/databamA --Train_set=m_TR.txt --Test_set=m_TE.txt --remove_mid=True --epochs_step=20 --_batch_size=128 --limit_v=1 --num_CV=5

python k_test_model5.py --data_path=./data/balance/databamA --Train_set=0m_TR.txt --Test_set=0m_TE.txt --remove_mid=True --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/balance/databamA --Train_set=1m_TR.txt --Test_set=1m_TE.txt --remove_mid=True --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/balance/databamA --Train_set=2m_TR.txt --Test_set=2m_TE.txt --remove_mid=True --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/balance/databamA --Train_set=3m_TR.txt --Test_set=3m_TE.txt --remove_mid=True --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/balance/databamA --Train_set=4m_TR.txt --Test_set=4m_TE.txt --remove_mid=True --epochs_step=20 --_batch_size=128 --limit_v=1

c)謎底版
python k_test_model6.py --data_path=./data/balance/databamB --Train_set=m_TR.txt --Test_set=m_TE.txt --remove_mid=True --epochs_step=20 --_batch_size=128 --limit_v=1 --num_CV=5

python k_test_model5.py --data_path=./data/balance/databamB --Train_set=0m_TR.txt --Test_set=0m_TE.txt --remove_mid=True --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/balance/databamB --Train_set=1m_TR.txt --Test_set=1m_TE.txt --remove_mid=True --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/balance/databamB --Train_set=2m_TR.txt --Test_set=2m_TE.txt --remove_mid=True --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/balance/databamB --Train_set=3m_TR.txt --Test_set=3m_TE.txt --remove_mid=True --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/balance/databamB --Train_set=4m_TR.txt --Test_set=4m_TE.txt --remove_mid=True --epochs_step=20 --_batch_size=128 --limit_v=1
