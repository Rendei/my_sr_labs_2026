# The script is borrowed from the following repository: https://github.com/clovaai/voxceleb_trainer
# The script downloads, extracts and preprocessing the VoxCeleb1 (train and test), SLR17 and SLR28 datasets 
# Requirement: wget running on a Linux system 


# Import of modules
import os
import time
import subprocess
import base64
import hashlib
import tarfile
import urllib.error
import urllib.request
from zipfile import ZipFile
import glob
import numpy as np
from tqdm import tqdm
from scipy.io import wavfile
from pathlib import Path
from multiprocessing import Pool
import soundfile
from scipy import signal
import random
from functools import partial


def check_dir(pth_):
    if not Path(pth_).is_dir():
        Path(pth_).mkdir(parents=True, exist_ok=True)


def md5(fname):
    """
    Estimate md5 sum
    """

    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    
    return hash_md5.hexdigest()


class _RedirectHandler(urllib.request.HTTPRedirectHandler):
    # Python < 3.11 does not follow 308 (permanent redirect), which is used by the VoxCeleb server
    http_error_308 = urllib.request.HTTPRedirectHandler.http_error_302

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # 308 behaves like 307: the method and the headers (e.g. Authorization, Range) are kept
        return super().redirect_request(req, fp, 307 if code == 308 else code, msg, headers, newurl)


_opener = urllib.request.build_opener(_RedirectHandler)


def download_file(url, out_path, user=None, password=None, max_tries=10, timeout=30, chunk_size=1 << 20):
    """
    Download a file with resuming (pure Python replacement of wget -c)
    :param url: http link
    :param out_path: path to output file
    :param user: user_name for basic authentication (optional)
    :param password: password for basic authentication (optional)
    :param max_tries: number of attempts before giving up
    :param timeout: socket timeout in seconds
    :param chunk_size: size of the chunk to read at once in bytes
    """

    headers = {}
    if user is not None:
        token = base64.b64encode(('%s:%s' % (user, password)).encode()).decode()
        headers['Authorization'] = 'Basic %s' % token

    part_path = out_path + '.part'
    for attempt in range(1, max_tries + 1):
        try:
            done = os.path.getsize(part_path) if os.path.exists(part_path) else 0
            request = urllib.request.Request(url, headers=dict(headers, Range='bytes=%d-' % done))

            with _opener.open(request, timeout=timeout) as response:
                if response.status == 206:  # server resumes the download
                    total = done + int(response.headers['Content-Length'])
                    mode = 'ab'
                else:                       # server sent the whole file
                    total = int(response.headers.get('Content-Length', 0))
                    done, mode = 0, 'wb'

                with open(part_path, mode) as f, tqdm(total=total or None, initial=done, unit='B',
                                                      unit_scale=True, desc=os.path.basename(out_path)) as bar:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        bar.update(len(chunk))

            if total and os.path.getsize(part_path) != total:
                raise IOError('Incomplete download: %d of %d bytes' % (os.path.getsize(part_path), total))

            os.replace(part_path, out_path)
            return

        except urllib.error.HTTPError as e:
            if e.code == 416:  # requested range is not satisfiable: .part file is already complete
                os.replace(part_path, out_path)
                return
            if e.code in (401, 403, 404):
                raise
            print('Download attempt %d/%d failed: %s' % (attempt, max_tries, e))
        except (urllib.error.URLError, IOError) as e:
            print('Download attempt %d/%d failed: %s' % (attempt, max_tries, e))

        time.sleep(5)

    raise IOError('Download failed %s after %d attempts.' % (url, max_tries))


def download_dataset(lines, user, password, save_path, reload=False):
    """
    Download datasets from lines
    :param lines: list of datasest to load in format <http_link> <md5 sum>\n
    :param user: user_name
    :param password:
    :param save_path: path to folder
    :param reload: rewrite if file exists
    """

    check_dir(save_path)

    for line in lines:
        url = line.strip().split(' ')[0]
        md5gt = line.strip().split(' ')[1]
        outfile = url.split('/')[-1]

        # Download files if needed
        if not os.path.exists(os.path.join(save_path, outfile)) or reload:
            download_file(url, os.path.join(save_path, outfile), user=user, password=password)

        # Check MD5
        md5ck = md5('%s/%s'%(save_path, outfile))
        if md5ck == md5gt:
            print('Checksum successful %s.'%outfile)
        
        else:
            raise Warning('Checksum failed %s.'%outfile)


def concatenate(lines, save_path):
    # Concatenate file parts

    for line in lines:
        infile  = line.split()[0]
        outfile = line.split()[1]
        md5gt   = line.split()[2]

        # Concatenate files
        out = subprocess.call('cat %s/%s > %s/%s' %(save_path, infile, save_path, outfile), shell=True)

        # Check MD5
        md5ck = md5('%s/%s'%(save_path, outfile))
        if md5ck == md5gt:
            print('Checksum successful %s.'%outfile)
        
        else:
            raise Warning('Checksum failed %s.'%outfile)

        out = subprocess.call('rm %s/%s' %(save_path, infile), shell=True)


def download_protocol(lines, save_path, reload=False):
    # Download with wget

    for line in lines:
        url     = line.strip()
        outfile = url.split('/')[-1]

        out = 0
        # Download files
        if not os.path.exists(os.path.join(save_path, outfile)) or reload:
            out = subprocess.call('wget %s -O %s/%s'%(url, save_path, outfile), shell=True)
        
        if out != 0:
            raise ValueError('Download failed %s. If download fails repeatedly, use alternate URL on the VoxCeleb website.'%url)

        print('File %s is downloaded.'%outfile)


def extract_dataset(save_path, fname):
    # Extract zip files
    
    if fname.endswith(".tar.gz"):
        
        with tarfile.open(fname, "r:gz") as tar:
            tar.extractall(save_path)
    
    elif fname.endswith(".zip"):
        
        with ZipFile(fname, 'r') as zf:
            zf.extractall(save_path)

    print('Extracting of %s is successful.'%fname)


def part_extract(save_path, fname, target):
    # Partially extract zip files
    
    print('Extracting %s'%fname)
    
    with ZipFile(fname, 'r') as zf:
        
        for infile in zf.namelist():
            
            if any([infile.startswith(x) for x in target]):
                zf.extract(infile, save_path)


def split_musan(save_path):
    # Split MUSAN (SLR17) dataset for faster random access
    
    files = glob.glob('%s/musan/*/*/*.wav'%save_path)

    audlen = 16000*5
    audstr = 16000*3

    for idx,file in enumerate(files):
        fs,aud = wavfile.read(file)
        writedir = os.path.splitext(file.replace('/musan/','/musan_split/'))[0]
        
        os.makedirs(writedir)
        
        for st in range(0,len(aud)-audlen, audstr):
            wavfile.write(writedir+'/%05d.wav'%(st/fs), fs, aud[st:st+audlen])

        print(idx,file)


def run_voxceleb_convert(input_path, result_path, fun, threads=5):
    """
    run files processing in a parralel mode using function fun
    """

    check_dir(result_path)
    infile_list = []
    outfile_list = []
    print('Checking result folder')

    for root, dir_list, files_list in os.walk(input_path):
        for i in dir_list:
            check_dir(Path(root.replace(input_path, result_path))/i)
        for f in files_list:
            infile = Path(root)/f
            infile_list.append(infile)
            outfile = Path(root.replace(input_path, result_path))/f.replace('.m4a', '.wav')
            outfile_list.append(outfile)

    p = Pool(threads)
    print('Run {} folder processing'.format(input_path))
    p.starmap(fun, zip(infile_list, outfile_list))
    p.close()
    p.join()

    print('Finished {} folder processing'.format(input_path))


def get_voxceleb_filelist(input_path):
    """
    run files processing in a parralel mode using function fun
    """
    infile_list = []
    for root, dir_list, files_list in os.walk(input_path):
        for f in files_list:
            infile = Path(root.replace(input_path + '/', ''))/f
            infile_list.append(str(infile))
    return infile_list


def aac_to_wav(infile, outfile):
    out = subprocess.call(
        'ffmpeg -y -i %s -ac 1 -vn -acodec pcm_s16le -ar 16000 %s >/dev/null 2>/dev/null' % (infile, outfile),
        shell=True)
    if out != 0:
        raise ValueError('Conversion failed %s.' % infile)


def change_fs(infile, outfile, fs):
    out = subprocess.call('ffmpeg -i %s -ar %s %s >/dev/null 2>/dev/null' % (infile, fs, outfile), shell=True)
    if out != 0:
        raise ValueError('Conversion failed %s.' % infile)


def apply_mp3_codec(infile, outfile):
    out = subprocess.call(
        'ffmpeg -i %s -f mp3 -ac 1 -b:a 16k pipe: 2>/dev/null | ffmpeg -f mp3 -i pipe: %s >/dev/null 2>/dev/null' % (
        infile, outfile), shell=True)
    if out != 0:
        raise ValueError('Conversion failed %s.' % infile)


def convert_16_8_16(infile, outfile):
    out = subprocess.call('ffmpeg -i %s -f s16le -ar 8000 pipe: 2>/dev/null | ffmpeg -f s16le -ar 8000 -i pipe: -ar 16000 %s >/dev/null 2>/dev/null'% (infile, outfile), shell=True)
    if out != 0:
        raise ValueError('Conversion failed %s.' % infile)


def reverberate(infile, outfile, rir_files):
    # print('Processing {}'.format(infile))
    rir_file = random.choice(rir_files)
    audio, fs_a = soundfile.read(infile)
    rir, fs_r = soundfile.read(rir_file)
    rir = rir / np.sqrt(np.sum(rir ** 2))
    audio_new = signal.convolve(audio, rir, mode='full')
    soundfile.write(outfile, audio_new, fs_a)
    return

