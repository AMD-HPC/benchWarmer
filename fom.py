import pdb
import os
import subprocess as sp
import sys

def get_gpu_name():
    """Fetch GPU name using rocminfo"""
    gpu_name = sp.run(
        'rocminfo | grep "Marketing Name" | tail -n 1 | cut -d ":" -f 2',
        shell=True,
        stdout=sp.PIPE,
        stderr=sp.PIPE,
        universal_newlines=True,
    )
    if gpu_name.stdout.strip() == '':
        print_error(f'GPU Name not resolved. Try using the flag: "-gname/--gpu_name"')
        raise Exception(f'GPU Name not found. Found "{gpu_name.stdout.strip()}"')

    return gpu_name.stdout.split()[2].upper().split('/')[0]

def read_file(file_name):
    try:
        with open(file_name, 'r') as file:
            content = file.read()
    except FileNotFoundError:
        print(f"The file {file_name} does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")
    tests = content.split('\n')
    tests = tests[1:-1]  # Remove first line (header) and last line (empty)

    gpuName = get_gpu_name()

    # generate `fomsend.sh` script
    fomsend_script = os.path.abspath('./fomsend.sh')
    with open(fomsend_script, 'w') as f:
        f.write('#!/bin/bash\n')
        # command to gather system information
        f.write('eval "./sys-config.sh -a benchwarmer"\n')

    # publish each line of output as a separate metric
    for test in tests:
        fom = test.split(',')
        send_cmd = f'./log-stats.sh benchwarmer {fom[0]}-{fom[1].strip()} {fom[2]} 1 "std={fom[3].strip()},nFlops={fom[5].strip()},nBytesAccessed={fom[6].strip()},nBlocks={fom[8].strip()},nThreads={fom[9].strip()},nExperiments={fom[10].strip()},gpuName={gpuName}" 0 0'

        # append new command to `fomsend.sh` script
        with open(fomsend_script, 'a') as f:
            f.write(send_cmd + '\n')

    # make script executable and run
    os.chmod(fomsend_script, os.stat(fomsend_script).st_mode | 0o111)
    publish_log = os.path.join('.', f'benchwarmer_publish.log')

    with open(publish_log, 'w') as f:
        sp.run(['bash', fomsend_script], stdout=f, stderr=f)

    curl_script = os.path.abspath('./curl-cmds.sh')
    os.chmod(curl_script, os.stat(curl_script).st_mode | 0o111)
    with open(publish_log, 'a') as f:
        sp.run(['bash', curl_script], stdout=f, stderr=f)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <filename>")
    read_file(sys.argv[1])

