# New
�N��l��ƶi��w�B�z
1. �u�O�d����r
2. �������I�Ÿ�
3. ��Ʈ榡[���O, ����, ����]
4. ��X��pkl��
�{�� 01_clean_data.py
�Ѽ�
--data_path�A��ƪ��m�M��ƪ�A�w�]".\data\newProverb2_v2_0.csv"
--output_path�A����X��ƪ���m
--dict_path�AJieba�_���һݪ����w�A�w�]�ϥ�Jieba�w�]���w�A�B�~���w�s���".\tools\extra_dict\dict.txt"

python 0_clean_data.py --data_path=./data/newProverb2_v2_0.csv --output_path=./pkl_data/newProverb2_jieba_v2_1.pkl --dict_path=./tools/extra_dict/dict.txt

�V�m�r���V�q
�{�� 13_train_word2vec.py
--dim_size�A�w�]=100�A�V�m�X�����V�q�|�����׼�
-sg�A�w�]=1�Asg=1��ܱĥ�skip-gram,sg=0 ��ܱĥ�cbow
--workers�A�w�]=4�A��������ƥ�
--min_count�A�w�]=5�A�Y�o�ӵ��X�{�����Ƥp��min_count�A���L�N���|�Q�����V�m��H
--window�A�w�]=5�A���R�r�������f�j�p
--sentences_path�A�����R�奻��ƪ����|��m
--output_path�A����X��ƪ���m

python 13_train_word2vec.py --dim_size=100 -sg=1 --workers=4 --min_count=1 --window=5 --sentences_path=./word2vec_data/segmentation.txt --output_path=./word2vec
python 13_train_word2vec.py --dim_size=300 -sg=1 --workers=4 --min_count=1 --window=5 --sentences_path=segmentation2.txt --output_path=.

# Old
Dataset PreProcessing
�����ƶ����Φ�������
�{�� datasetCV.py
�Ѽ�
--analysis_type�A���Τ��R��������All�BA�BB�T�ءAAll�������M�����Ҧ��AA�������AB������
--data_path�A��ƪ��m�M��ƪ�A�w�]"./Proverb_nonexpansion_all_seq2seq_v1.csv"
--output_path�A����X��ƪ���m
--num_CV�A��e���Ҫ������ƶq�A�w�]��5

�N�V�m�M���ժ���r����ഫ���r���V�q(�H�V�m��ƶ����̾ڡA�N���ն��ഫ������)
�{�� convert_wordvector.py
�Ѽ�
--analysis_type�A���R�����A�P�W�z�ѼƬۦP�N��C
--data_path�A��e���Ҹ�ƪ���m
--output_path�A����X��ƪ���m
--num_CV�A��e���Ҫ��ƶq�A�w�]��5
--remove_mid�A�O�_�����������O�A�w�]False
--equallabel�A�H�̤p���O���ƶq���̾ڡA�w�]False

python convert_wordvector.py --data_path=./res --output_path=./res --analysis_type=All --n=None --train_filename=0m_TR --test_filename=0m_TE

A. �L�����������O�B�L���ä������O����
�N�����ư������e����
python datasetCV.py --data_path=./Proverb_nonexpansion_all_seq2seq_v1.csv --output_path=./datas --remove_mid=False --equallabel=False --num_CV=5
1.  �����M������
python convert_wordvector.py --data_path=./datas --output_path=./data --analysis_type=All --num_CV=5
2.  ������
python convert_wordvector.py --data_path=./datas --output_path=./dataA --analysis_type=A --num_CV=5
3.  ������
python convert_wordvector.py --data_path=./datas --output_path=./dataB --analysis_type=B --num_CV=5

B. �L�����������O�B�����ä������O����
python datasetCV.py --data_path=./Proverb_nonexpansion_all_seq2seq_v1.csv --output_path=./datasb --remove_mid=False --equallabel=True --num_CV=5

python convert_wordvector.py --data_path=./datasb --output_path=./databa --analysis_type=All --num_CV=5

python convert_wordvector.py --data_path=./datasb --output_path=./databaA --analysis_type=A --num_CV=5

python convert_wordvector.py --data_path=./datasb --output_path=./databaB --analysis_type=B --num_CV=5

C. �������������O�B�L���ä������O����
python datasetCV.py --data_path=./Proverb_nonexpansion_all_seq2seq_v1.csv --output_path=./datasm --remove_mid=True --equallabel=False --num_CV=5

python convert_wordvector.py --data_path=./datasm --output_path=./datam --analysis_type=All --num_CV=5

python convert_wordvector.py --data_path=./datasm --output_path=./datamA --analysis_type=A --num_CV=5

python convert_wordvector.py --data_path=./datasm --output_path=./datamB --analysis_type=B --num_CV=5

D. �������������O�B�����ä������O����
python datasetCV.py --data_path=./Proverb_nonexpansion_all_seq2seq_v1.csv --output_path=./datasbm --remove_mid=True --equallabel=True --num_CV=5

python convert_wordvector.py --data_path=./datasbm --output_path=./databam --analysis_type=All --num_CV=5

python convert_wordvector.py --data_path=./datasbm --output_path=./databamA --analysis_type=A --num_CV=5

python convert_wordvector.py --data_path=./datasbm --output_path=./databamB --analysis_type=B --num_CV=5

----------------------------------------------------------------------------------------------------------------------------------------------
RUN Classifier
�����ҫ�
�{�� k_test_model5.py
�Ѽ�
--data_path�A����ƪ���m�A�̫�Фť[�W�׽u�A�d�ҡG./data/nonbalance/data
--Train_set�A�V�m��ƶ����ɦW�M���ɦW�A0m_TR.txt
--Test_set�A���ո�ƶ����ɦW�M���ɦW0m_TE.txt
--remove_mid�A�O�_�����������O�A�w�]False
--epochs_step�A�^�N���ơA�w�]20
--_batch_size�A�妸���ƶq�A�w�]128
--limit_v�A�w��r���V�q���W�v�A�]�w1��1�H�W���ҫO�d�C

A. �L�����������O�B�L���ä������O����
a)�����M������
python k_test_model6.py --data_path=./data/nonbalance/data --Train_set=m_TR.txt --Test_set=m_TE.txt --remove_mid=False --epochs_step=20 --_batch_size=128 --limit_v=1 --num_CV=5

python k_test_model5.py --data_path=./data/nonbalance/data --Train_set=0m_TR.txt --Test_set=0m_TE.txt --remove_mid=False --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/nonbalance/data --Train_set=1m_TR.txt --Test_set=1m_TE.txt --remove_mid=False --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/nonbalance/data --Train_set=2m_TR.txt --Test_set=2m_TE.txt --remove_mid=False --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/nonbalance/data --Train_set=3m_TR.txt --Test_set=3m_TE.txt --remove_mid=False --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/nonbalance/data --Train_set=4m_TR.txt --Test_set=4m_TE.txt --remove_mid=False --epochs_step=20 --_batch_size=128 --limit_v=1

b)������
python k_test_model6.py --data_path=./data/nonbalance/dataA --Train_set=m_TR.txt --Test_set=m_TE.txt --remove_mid=False --epochs_step=20 --_batch_size=128 --limit_v=1 --num_CV=5

python k_test_model5.py --data_path=./data/nonbalance/dataA --Train_set=0m_TR.txt --Test_set=0m_TE.txt --remove_mid=False --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/nonbalance/dataA --Train_set=1m_TR.txt --Test_set=1m_TE.txt --remove_mid=False --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/nonbalance/dataA --Train_set=2m_TR.txt --Test_set=2m_TE.txt --remove_mid=False --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/nonbalance/dataA --Train_set=3m_TR.txt --Test_set=3m_TE.txt --remove_mid=False --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/nonbalance/dataA --Train_set=4m_TR.txt --Test_set=4m_TE.txt --remove_mid=False --epochs_step=20 --_batch_size=128 --limit_v=1

c)������
python k_test_model6.py --data_path=./data/nonbalance/dataB --Train_set=m_TR.txt --Test_set=m_TE.txt --remove_mid=False --epochs_step=20 --_batch_size=128 --limit_v=1 --num_CV=5

python k_test_model5.py --data_path=./data/nonbalance/dataB --Train_set=0m_TR.txt --Test_set=0m_TE.txt --remove_mid=False --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/nonbalance/dataB --Train_set=1m_TR.txt --Test_set=1m_TE.txt --remove_mid=False --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/nonbalance/dataB --Train_set=2m_TR.txt --Test_set=2m_TE.txt --remove_mid=False --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/nonbalance/dataB --Train_set=3m_TR.txt --Test_set=3m_TE.txt --remove_mid=False --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/nonbalance/dataB --Train_set=4m_TR.txt --Test_set=4m_TE.txt --remove_mid=False --epochs_step=20 --_batch_size=128 --limit_v=1

B. �L�����������O�B�����ä������O����
a)�����M������
python k_test_model6.py --data_path=./data/balance/databa --Train_set=m_TR.txt --Test_set=m_TE.txt --remove_mid=False --epochs_step=20 --_batch_size=128 --limit_v=1 --num_CV=5

python k_test_model5.py --data_path=./data/balance/databa --Train_set=0m_TR.txt --Test_set=0m_TE.txt --remove_mid=False --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/balance/databa --Train_set=1m_TR.txt --Test_set=1m_TE.txt --remove_mid=False --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/balance/databa --Train_set=2m_TR.txt --Test_set=2m_TE.txt --remove_mid=False --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/balance/databa --Train_set=3m_TR.txt --Test_set=3m_TE.txt --remove_mid=False --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/balance/databa --Train_set=4m_TR.txt --Test_set=4m_TE.txt --remove_mid=False --epochs_step=20 --_batch_size=128 --limit_v=1

b)������
python k_test_model6.py --data_path=./data/balance/databaA --Train_set=m_TR.txt --Test_set=m_TE.txt --remove_mid=False --epochs_step=20 --_batch_size=128 --limit_v=1 --num_CV=5

python k_test_model5.py --data_path=./data/balance/databaA --Train_set=0m_TR.txt --Test_set=0m_TE.txt --remove_mid=False --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/balance/databaA --Train_set=1m_TR.txt --Test_set=1m_TE.txt --remove_mid=False --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/balance/databaA --Train_set=2m_TR.txt --Test_set=2m_TE.txt --remove_mid=False --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/balance/databaA --Train_set=3m_TR.txt --Test_set=3m_TE.txt --remove_mid=False --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/balance/databaA --Train_set=4m_TR.txt --Test_set=4m_TE.txt --remove_mid=False --epochs_step=20 --_batch_size=128 --limit_v=1

c)������
python k_test_model6.py --data_path=./data/balance/databaB --Train_set=m_TR.txt --Test_set=m_TE.txt --remove_mid=False --epochs_step=20 --_batch_size=128 --limit_v=1 --num_CV=5

python k_test_model5.py --data_path=./data/balance/databaB --Train_set=0m_TR.txt --Test_set=0m_TE.txt --remove_mid=False --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/balance/databaB --Train_set=1m_TR.txt --Test_set=1m_TE.txt --remove_mid=False --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/balance/databaB --Train_set=2m_TR.txt --Test_set=2m_TE.txt --remove_mid=False --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/balance/databaB --Train_set=3m_TR.txt --Test_set=3m_TE.txt --remove_mid=False --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/balance/databaB --Train_set=4m_TR.txt --Test_set=4m_TE.txt --remove_mid=False --epochs_step=20 --_batch_size=128 --limit_v=1

C. �������������O�B�L���ä������O����
a)�����M������
python k_test_model6.py --data_path=./data/nonbalance/datam --Train_set=m_TR.txt --Test_set=m_TE.txt --remove_mid=True --epochs_step=20 --_batch_size=128 --limit_v=1 --num_CV=5

python k_test_model5.py --data_path=./data/nonbalance/datam --Train_set=0m_TR.txt --Test_set=0m_TE.txt --remove_mid=True --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/nonbalance/datam --Train_set=1m_TR.txt --Test_set=1m_TE.txt --remove_mid=True --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/nonbalance/datam --Train_set=2m_TR.txt --Test_set=2m_TE.txt --remove_mid=True --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/nonbalance/datam --Train_set=3m_TR.txt --Test_set=3m_TE.txt --remove_mid=True --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/nonbalance/datam --Train_set=4m_TR.txt --Test_set=4m_TE.txt --remove_mid=True --epochs_step=20 --_batch_size=128 --limit_v=1

b)������
python k_test_model6.py --data_path=./data/nonbalance/datamA --Train_set=m_TR.txt --Test_set=m_TE.txt --remove_mid=True --epochs_step=20 --_batch_size=128 --limit_v=1 --num_CV=5

python k_test_model5.py --data_path=./data/nonbalance/datamA --Train_set=0m_TR.txt --Test_set=0m_TE.txt --remove_mid=True --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/nonbalance/datamA --Train_set=1m_TR.txt --Test_set=1m_TE.txt --remove_mid=True --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/nonbalance/datamA --Train_set=2m_TR.txt --Test_set=2m_TE.txt --remove_mid=True --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/nonbalance/datamA --Train_set=3m_TR.txt --Test_set=3m_TE.txt --remove_mid=True --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/nonbalance/datamA --Train_set=4m_TR.txt --Test_set=4m_TE.txt --remove_mid=True --epochs_step=20 --_batch_size=128 --limit_v=1

c)������
python k_test_model6.py --data_path=./data/nonbalance/datamB --Train_set=m_TR.txt --Test_set=m_TE.txt --remove_mid=True --epochs_step=20 --_batch_size=128 --limit_v=1 --num_CV=5

python k_test_model5.py --data_path=./data/nonbalance/datamB --Train_set=0m_TR.txt --Test_set=0m_TE.txt --remove_mid=True --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/nonbalance/datamB --Train_set=1m_TR.txt --Test_set=1m_TE.txt --remove_mid=True --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/nonbalance/datamB --Train_set=2m_TR.txt --Test_set=2m_TE.txt --remove_mid=True --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/nonbalance/datamB --Train_set=3m_TR.txt --Test_set=3m_TE.txt --remove_mid=True --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/nonbalance/datamB --Train_set=4m_TR.txt --Test_set=4m_TE.txt --remove_mid=True --epochs_step=20 --_batch_size=128 --limit_v=1

D. �������������O�B�����ä������O����
a)�����M������
python k_test_model6.py --data_path=./data/balance/databam --Train_set=m_TR.txt --Test_set=m_TE.txt --remove_mid=True --epochs_step=20 --_batch_size=128 --limit_v=1 --num_CV=5

python k_test_model5.py --data_path=./data/balance/databam --Train_set=0m_TR.txt --Test_set=0m_TE.txt --remove_mid=True --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/balance/databam --Train_set=1m_TR.txt --Test_set=1m_TE.txt --remove_mid=True --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/balance/databam --Train_set=2m_TR.txt --Test_set=2m_TE.txt --remove_mid=True --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/balance/databam --Train_set=3m_TR.txt --Test_set=3m_TE.txt --remove_mid=True --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/balance/databam --Train_set=4m_TR.txt --Test_set=4m_TE.txt --remove_mid=True --epochs_step=20 --_batch_size=128 --limit_v=1

b)������
python k_test_model6.py --data_path=./data/balance/databamA --Train_set=m_TR.txt --Test_set=m_TE.txt --remove_mid=True --epochs_step=20 --_batch_size=128 --limit_v=1 --num_CV=5

python k_test_model5.py --data_path=./data/balance/databamA --Train_set=0m_TR.txt --Test_set=0m_TE.txt --remove_mid=True --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/balance/databamA --Train_set=1m_TR.txt --Test_set=1m_TE.txt --remove_mid=True --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/balance/databamA --Train_set=2m_TR.txt --Test_set=2m_TE.txt --remove_mid=True --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/balance/databamA --Train_set=3m_TR.txt --Test_set=3m_TE.txt --remove_mid=True --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/balance/databamA --Train_set=4m_TR.txt --Test_set=4m_TE.txt --remove_mid=True --epochs_step=20 --_batch_size=128 --limit_v=1

c)������
python k_test_model6.py --data_path=./data/balance/databamB --Train_set=m_TR.txt --Test_set=m_TE.txt --remove_mid=True --epochs_step=20 --_batch_size=128 --limit_v=1 --num_CV=5

python k_test_model5.py --data_path=./data/balance/databamB --Train_set=0m_TR.txt --Test_set=0m_TE.txt --remove_mid=True --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/balance/databamB --Train_set=1m_TR.txt --Test_set=1m_TE.txt --remove_mid=True --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/balance/databamB --Train_set=2m_TR.txt --Test_set=2m_TE.txt --remove_mid=True --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/balance/databamB --Train_set=3m_TR.txt --Test_set=3m_TE.txt --remove_mid=True --epochs_step=20 --_batch_size=128 --limit_v=1
python k_test_model5.py --data_path=./data/balance/databamB --Train_set=4m_TR.txt --Test_set=4m_TE.txt --remove_mid=True --epochs_step=20 --_batch_size=128 --limit_v=1
