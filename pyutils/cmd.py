import subprocess
from subprocess import PIPE
import gc


def runSystemCMD(cmd):
    gc.collect()
    p = subprocess.Popen(cmd.split(), stdout=PIPE, stderr=PIPE, close_fds=True)
    stdout, stderr = p.communicate()
    if p.returncode != 0:
        raise ValueError, 'System call returned with error.\n CMD: {}\n STDOUT: {}\nSTDERR: {}'.format(cmd, stdout, stderr)

    p.stderr.close()
    p.stdout.close()
    p.wait()

    return stdout, stderr
