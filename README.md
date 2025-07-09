# cyber_bullying
python run_experiments.py --baseline > baseline_output.log 2>&1
python run_experiments.py --bert > bert_output.log 2>&1
python run_experiments.py --bilstm > bilstm_output.log 2>&1
python run_experiments.py --emotion > emotion_output.log 2>&1
python run_experiments.py --all > all_experiments_output.log 2>&1
OR just:
 python run_experiments.py > all_experiments_output.log 2>&1