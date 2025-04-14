import subprocess

processes = []

processes.append(subprocess.Popen(['python', 'colbert_retrieve.py 0 10000']))
processes.append(subprocess.Popen(['python', 'colbert_retrieve.py 10000 20000']))
processes.append(subprocess.Popen(['python', 'colbert_retrieve.py 20000 30000']))
processes.append(subprocess.Popen(['python', 'colbert_retrieve.py 30000 40000']))

for process in processes:
    process.wait()