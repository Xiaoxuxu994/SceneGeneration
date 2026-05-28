# encoding: utf-8
# Author: Tao yuheng(t50018193).
import argparse
import base64
import copy
import json
import logging
import os
import platform
import queue
import ssl
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from urllib import request

import requests

os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)

logging.basicConfig(level=logging.ERROR, format='[%(asctime)s] [%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

endpoint = "https://csb.roma.huawei.com"
api = "/csb/rest/s3/bucket/endpoint"

parser = argparse.ArgumentParser(description='Parameters of CSB-OBS',
                                 formatter_class=
                                 argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--show_speed", action='store_true',
                    help="To display the current download speed, <psutil> is required")
parser.add_argument("--single_file", action='store_true',
                    help="if you download single file please add '--single_file' behind your cmd")
parser.add_argument("--bucket_path", type=str, required=True,
                    help="The default value of bucket path")
parser.add_argument('--region', type=str, required=True,
                    help='region of bucket.')
parser.add_argument('--bucket_name', type=str, required=True,
                    help='Please input you bucket_name.')
parser.add_argument('--bucket_id', type=str, required=False,
                    help='Please input you bucket_id.')
parser.add_argument('--app_token', type=str, required=True,
                    help='appToken of CSB.')
parser.add_argument('--objects_storage_path', type=str, required=True,
                    help='Output list of objectkeys.')
parser.add_argument('--buffer_size', type=int, default=65536,
                    help='The default value of buffer_size is 65536.')
parser.add_argument("--retry_times", type=int, default=3,
                    help='The default value of retry times.')
parser.add_argument('--big_file', type=int, default=100 * 1024 * 1024,
                    help='The default value of big file is 100M.')
parser.add_argument('--thread_num', type=int, default=12,
                    help='The default value of thread.')
parser.add_argument("--package_size", type=int, default=50 * 1024 * 1024,
                    help='The default value of package is 50M.')
parser.add_argument('--vendor', type=str, default="HEC",
                    help='vendor of bucket.')
parser.add_argument('--queue_size', type=int, default=50000,
                    help="max value of download_queue")
parser.add_argument("--fail_json_storage_path", type=str, default="./fail.json",
                    help="The default value of fail.json storage path")
parser.add_argument("--time_wait", type=int, default=10,
                    help="The default value of download thread waiting time (second)")
parser.add_argument("--debug", action='store_true',
                    help="open debug mode to get more system output")
parser.add_argument("--shared", action='store_true',
                    help="if the bucket is shared")
parser.add_argument("--include_folder", action='store_true',
                    help="if download empty folder")
parser.add_argument('--include_file_name', type=str, required=False, action="append",
                    help='indicates the matching patterns of source objects that are included, '
                         'such as \.jpg match all .jpg suffix file')
parser.add_argument('--exclude_file_name', type=str, required=False, action="append",
                    help='indicates the matching patterns of source objects that are excluded,'
                         ' such as \.jpg match all .jpg suffix file')
parser.add_argument("--incremental", action='store_true',
                    help="Indicates incremental download. If this parameter is set, "
                         "each object can be downloaded only when it does not exist "
                         "in the local path, its size is different from the namesake "  
                         "one in the local path.")
parser.add_argument('--version', type=str, default="2025-05-28", help='The script version')
parser.add_argument('--delimiter', type=str, default="/", help='Separator for grouping object names')
parser.add_argument('--bucket_type', type=str, default="OBJECT", help='the bucket type')

result, _ = parser.parse_known_args()
args = copy.deepcopy(result)
context = ssl._create_unverified_context()

if args.shared:
    assert args.bucket_id is not None, "must input bucket_id params in shared mode!"

include_pattern_list = []
exclude_pattern_list = []
if args.include_file_name is not None or args.exclude_file_name is not None:
    import re

    # 初始化
    if args.include_file_name is None:
        args.include_file_name = []
    if args.exclude_file_name is None:
        args.exclude_file_name = []

    for include_name in args.include_file_name:
        include_pattern_list.append(re.compile(include_name))
    for exclude_name in args.exclude_file_name:
        exclude_pattern_list.append(re.compile(exclude_name))

if args.bucket_path.startswith(args.bucket_name):
    raise Exception(f"bucket_path:[{args.bucket_path}] can not start with bucket_name:[{args.bucket_name}]")

if args.bucket_path.startswith("/"):
    raise Exception(f"bucket_path:[{args.bucket_path}] can not start with /")

if not args.single_file:
    assert args.bucket_path.endswith("/"), "bucket_path must ends with /"
    args.father_path = args.bucket_path.split("/")[-2] + "/"
    # do path transfer when downloading all bucket file
    if len(args.bucket_path) == 1 and args.bucket_path == "/":
        args.bucket_path = ""
else:
    args.father_path = args.bucket_path[args.bucket_path.rfind("/") + 1:]

if args.father_path == "/":
    args.father_path = ""

print_lock = threading.Lock()
file_lock = threading.Lock()
download_queue = queue.Queue(args.queue_size)
big_file_download_queue = queue.Queue()
range_queue = queue.Queue()
big_file_ls = []
file_count = 0
downloaded_count = 0
skip_count = 0
big_file_process_count = 0
big_file_skip_count = 0
big_file_process_total = 0
error_count = 0

current_speed = ""
start_count_speed = True

has_next = True

retry_map = defaultdict(int)
error_map = dict()
next_marker = ""

session = requests.Session()


# 0. get endpoint
def get_file_server_endpoint():
    try:
        url = endpoint + api
        param = "?" + "bucketid=" + args.bucket_name \
                + "&token="  + args.app_token \
                + "&vendor=" + args.vendor \
                + "&region=" + args.region \
                + '&version=' + args.version
        if args.shared:
            param = param + "&bucketUid=" + args.bucket_id
        req = request.Request(url=url + param)
        res = request.urlopen(req, context=context, timeout=60)
        result = res.read().decode(encoding='utf-8')
        result_dict = json.loads(result)
        if result_dict["success"]:
            # add version verification
            print("check version result: " + result_dict["msg"])
            print("endpoint is : " + result_dict["result"])
            args.bucket_type = get_bucket_type(result_dict)
            print_split()
            return result_dict["result"]
        else:
            print(result_dict["msg"])
            raise Exception("get endpoint failed")
    except Exception as e:
        print("get endpoint failed, please check params and network. For more infomation refer to https://wiki.huawei.com/domains/62117/wiki/90979/WIKI202401252833508")
        if args.debug:
            print("exception is: ", e)
        print_split()
        sys.exit()


# 1. bucket_auth
def bucket_auth():
    bucket_auth_endpoint = args.csb_file_server + \
                           '/rest/boto3/s3/bucket-auth?' \
                           'vendor=' + args.vendor \
                           + '&region=' + args.region \
                           + '&bucketid=' + args.bucket_name \
                           + '&apptoken=' + args.app_token
    if args.shared:
        bucket_auth_endpoint = bucket_auth_endpoint + "&bucketUid=" + args.bucket_id
        print(bucket_auth_endpoint)
    try:
        req = request.Request(url=bucket_auth_endpoint)
        res = request.urlopen(req, timeout=60)
        result = res.read().decode(encoding='utf-8')
        if args.debug:
            print("bucket auth result: " + result)
        result_dict = json.loads(result)
        if not result_dict["success"]:
            raise Exception(result_dict["msg"])
    except Exception as e:
        print("bucket auth failed, please check the token and network")
        print("url: " + bucket_auth_endpoint)
        if args.debug:
            print("exception is: ", e)
        print_split()
        sys.exit()


def human_size(size, dot=3):
    return str(round(size / pow(1024, 3), dot)) + ' GB'


# 2. get objectKey
def get_object_key(next_marker=""):
    global has_next
    global file_count
    path = "/rest/boto3/s3/list/bucket/objectkeys?"
    object_key = base64.urlsafe_b64encode(args.bucket_path.encode()).decode()
    param = f"vendor={args.vendor}&region={args.region}&bucketid={args.bucket_name}" \
            f"&apptoken={args.app_token}&objectkey={object_key}&nextmarker={next_marker}"
    if args.include_folder:
        param = param + "&includeFolder=true"
    if args.shared:
        param = param + "&bucketId=" + args.bucket_id
    headers = {
        "Content-Type": "application/json",
        "csb-token": args.app_token
    }
    result = session.get(args.csb_file_server + path + param, headers=headers, timeout=60)
    try:
        result_dict = json.loads(result.content)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("list object key error", result.content)
        print("exception is: ", e)
        raise Exception("list object key error")

    if not result_dict["nextmarker"]:
        has_next = False
    if not result_dict["success"]:
        logging.error(result.text)
        error_map["next_marker" + "@" + next_marker] = result.text
        raise Exception("get object key failed")
    file_count += len(result_dict["objectKeys"])
    for object_key in result_dict["objectKeys"]:

        include_flag = True
        exclude_flag = False
        if args.include_file_name is not None or args.exclude_file_name is not None:
            for include_pattern in include_pattern_list:
                # program need to download specific file when include_pattern_list is not none
                # change include_flag to false at first
                include_flag = False
                if include_pattern.search(object_key["objectKey"]) is not None:
                    # current object_key need to include
                    include_flag = True
                    break
            for exclude_pattern in exclude_pattern_list:
                if exclude_pattern.search(object_key["objectKey"]) is not None:
                    # current object_key need to exclude
                    exclude_flag = True
                    break

        if not include_flag or exclude_flag:
            file_count -= 1
            continue

        if int(object_key["size"]) > args.big_file:
            big_file_download_queue.put(object_key)
            file_count -= 1
        else:
            download_queue.put(object_key)
        if args.single_file:
            break
    return result_dict


def do_download_queue():
    global file_count
    result_dict = get_object_key()
    while result_dict["nextmarker"]:
        try:
            result_dict = get_object_key(result_dict["nextmarker"])
        except Exception:
            error_map[result_dict["nextmarker"]] = time.time()
            time.sleep(5)
            result_dict = get_object_key(result_dict["nextmarker"])


def get_object(object_name: str, object_size: int):
    global error_count
    global skip_count
    if not args.single_file:
        object_name_path = os.path.join(args.objects_storage_path, args.father_path,
                                        object_name.replace(args.bucket_path, ""))
        index = object_name_path.rfind("/")
        local_objects_storage_path = os.path.join(args.objects_storage_path, args.father_path) if index == -1 else object_name_path[:object_name_path.rfind("/")]
    else:
        object_name_path = os.path.join(args.objects_storage_path, args.father_path)
        local_objects_storage_path = args.objects_storage_path
    if args.incremental:
        if os.path.exists(object_name_path) and int(object_size) == int(os.path.getsize(object_name_path)):
            skip_count += 1
            print("\n--Skip <{}> size: {} byte".format(object_name, object_size))
            return
    with print_lock:
        if not os.path.exists(local_objects_storage_path):
            os.makedirs(local_objects_storage_path)
    b64_object_name = base64.urlsafe_b64encode(object_name.encode()).decode()
    # Not need to write empty folder
    if args.include_folder and object_name_path.endswith("/") and object_size == 0:
        return
    path = f"/rest/boto3/s3/{args.vendor}/{args.region}/{args.app_token}/{args.bucket_name}/{b64_object_name}"
    if args.shared:
        path = path + "?bucketId=" + args.bucket_id
    headers = {
        "Content-Type": "application/json",
        "csb-token": args.app_token,
        'Connection': 'close'
    }

    with session.get(args.csb_file_server + path, headers=headers, timeout=60) as result:
        if result.status_code > 200:
            raise Exception(
                f"download file error. return code {result.status_code}, exception:{result.content.decode('utf-8')} ")
        with open(object_name_path, "wb") as f:
            for content in result.iter_content(chunk_size=args.buffer_size):
                f.write(content)
            current_location = f.tell()
            if int(object_size) != int(current_location):
                raise Exception(
                    f"download file with wrong size. "
                    f"local file size: {current_location}, "
                    f"remote file size: {object_size}")


def get_range_object(object_name: str, object_size: int, h_size, file_io):
    global big_file_process_count
    global big_file_process_total
    global start_count_speed
    global current_speed
    b64_object_name = base64.urlsafe_b64encode(object_name.encode()).decode()
    # object_size = get_object_size(b64_object_name)
    while range_queue.qsize() > 0:
        content_range = range_queue.get()
        path = f"/rest/boto3/s3/{args.vendor}/{args.region}/{args.app_token}/{args.bucket_name}/{b64_object_name}"
        if args.shared:
            path = path + "?bucketId=" + args.bucket_id
        headers = {"Content-Type": "application/json",
                   "csb-token": args.app_token,
                   "Range": f"Range={content_range[0]}-{content_range[1]}",
                   "Date": datetime.utcnow().strftime('%a, %d %b %Y %H:%M:%S GMT'),
                   "Connection": "Keep-Alive"}
        try:
            with session.get(args.csb_file_server + path, headers=headers, timeout=60) as result:
                if result.status_code > 200:
                    raise Exception(
                        f"download file error. "
                        f"return code {result.status_code}, "
                        f"exception:{result.content.decode('utf-8')} ")
                with file_lock:
                    file_io.seek(content_range[0])
                    before_location = file_io.tell()
                    for content in result.iter_content(chunk_size=args.buffer_size):
                        file_io.write(content)
                    current_location = file_io.tell()
                    # when the file size is larger than local file size
                    if int(object_size) > int(current_location):
                        # current_location is the location of writing idx
                        # need to minus 1 to get size
                        size = current_location - before_location - 1
                        if size != content_range[1] - content_range[0]:
                            raise Exception(
                                f"download part file with wrong size. "
                                f"before file location: {before_location}, "
                                f"current file location: {current_location}, "
                                f"file content start: {content_range[0]}, "
                                f"file content end: {content_range[1]}")

            big_file_process_count += 1
            with print_lock:
                if big_file_process_count > big_file_process_total:
                    big_file_process_count = big_file_process_total
                print("\r", end="")
                print("Downloading <{}> size: {}: {}/{} - {}%: ".format(object_name, h_size,
                                                                        big_file_process_count, big_file_process_total,
                                                                        int((
                                                                                    big_file_process_count / big_file_process_total) * 100)),
                      "▋" * (int(big_file_process_count / big_file_process_total * 50)) + current_speed, end="")
                sys.stdout.flush()
        except Exception:
            import traceback
            traceback.print_exc()
            retry_map[str(content_range)] += 1
            if retry_map[str(content_range)] > args.retry_times:
                # log
                print(f"{content_range} try {args.retry_times} times failed")
                error_map[str(content_range)] = traceback.format_exc()
            else:
                time.sleep(30)
                range_queue.put(content_range)
                get_range_object(object_name, object_size, h_size, file_io)
    start_count_speed = False


def do_range_object(pool: ThreadPoolExecutor):
    global big_file_process_count
    global big_file_process_total
    global big_file_skip_count
    global start_count_speed
    while big_file_download_queue.qsize() > 0:
        big_file_process_count = 0
        download_content = big_file_download_queue.get()
        total_size = int(download_content["size"])
        object_name = download_content["objectKey"]

        if not args.single_file:
            object_name_path = os.path.join(args.objects_storage_path, args.father_path,
                                            object_name.replace(args.bucket_path, ""))
            local_objects_storage_path = object_name_path[:object_name_path.rfind("/")]
        else:
            object_name_path = os.path.join(args.objects_storage_path, args.father_path)
            local_objects_storage_path = args.objects_storage_path

        if args.incremental:
            if os.path.exists(object_name_path) and int(total_size) == int(os.path.getsize(object_name_path)):
                big_file_skip_count += 1
                print("\n--Skip <{}> size: {}".format(object_name, human_size(total_size)))
                sys.stdout.flush()
                print_split()
                continue

        with print_lock:
            if not os.path.exists(local_objects_storage_path):
                os.makedirs(local_objects_storage_path)

        one_thread_range = args.package_size
        range_list = [[i if i == 0 else i + 1, i + one_thread_range] for i in
                      range(0, total_size + 1, one_thread_range)]
        big_file_process_total = len(range_list)
        for r in range_list:
            range_queue.put(r)

        start_count_speed = True
        thread_count_current_speed = pool.submit(get_delta)

        file_io = open(object_name_path, "wb+")

        ls_thread = [pool.submit(get_range_object, object_name, total_size,
                                 human_size(total_size), file_io) for _ in range(args.thread_num)]
        ls_thread.append(thread_count_current_speed)
        for task in as_completed(ls_thread):
            try:
                task.result()
            except Exception as e:
                raise e
        file_io.close()
        print_split()


def do_get():
    global file_count
    global downloaded_count
    global current_speed
    global start_count_speed
    while download_queue.qsize() > 0 or has_next:
        try:
            download_content = download_queue.get(timeout=args.time_wait)
        except Exception:
            break
        object_key = download_content["objectKey"]
        object_size = int(download_content["size"])
        try:
            get_object(object_key, object_size)
            downloaded_count += 1
            with print_lock:
                print("\r", end="")
                print("Download progress: {}/{} - {}%: ".format(downloaded_count, file_count,
                                                                int((downloaded_count / file_count) * 100)),
                      "▋" * (int(downloaded_count / file_count * 50)) + current_speed, end="")
                sys.stdout.flush()
        except Exception:
            import traceback
            traceback.print_exc()

            retry_map[str(download_content)] += 1
            if retry_map[str(download_content)] > args.retry_times:
                logging.error(f"{download_content} try {args.retry_times} times failed")
                error_map[str(download_content)] = traceback.format_exc()
                continue
            download_queue.put(download_content)
    start_count_speed = False


def multi_thread_main():
    args.csb_file_server = get_file_server_endpoint()
    bucket_auth()
    print('begin downloading small file ...')
    full_thread_num = args.thread_num + 4
    pool = ThreadPoolExecutor(max_workers=full_thread_num, thread_name_prefix="Python Downloader")

    ls_thread = []
    thread_do_download_queue = pool.submit(do_download_queue_by_bucket_type)
    thread_count_current_speed = pool.submit(get_delta)
    ls_thread.append(thread_do_download_queue)
    for i in range(args.thread_num):
        ls_thread.append(pool.submit(do_get))
    ls_thread.append(thread_count_current_speed)
    tic = time.time()
    for task in as_completed(ls_thread):
        try:
            task.result()
        except Exception as e:
            raise e

    toc = time.time()
    print()
    seconds = toc - tic
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    print("small file download completed, cost time : {:0>d} h {:0>2d} m {:0>2d} s \n".format(int(h), int(m), int(s)))
    if args.incremental:
        print("skip small file number: {}".format(skip_count))

    print(f'begin downloading {big_file_download_queue.qsize()} big files ...')

    tic = time.time()
    do_range_object(pool)
    toc = time.time()
    seconds = toc - tic
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    print("big file download completed, cost time : {:0>d} h {:0>2d} m {:0>2d} s".format(int(h), int(m), int(s)))
    if args.incremental:
        print("skip big file number: {}".format(big_file_skip_count))
    print_split()


def print_split():
    print("\n-------------------------------------------\n")


def check_retry():
    print(f"error_count =  {error_count}")
    if error_map:
        with open(args.fail_json_storage_path, "w", encoding="utf-8") as f:
            json.dump(error_map, f)
        logging.error(f"{len(error_map)} files download failed, written in {args.fail_json_storage_path}")
    else:
        print("all files download completed!")
    if "Windows" in platform.platform():
        set_display_required(False)


def check_platform():
    if "Windows" in platform.platform():
        print("检测到使用Windows系统，在Windows系统下，该脚本会阻止系统休眠，以确保下载能够完成。", end='')
        set_display_required(True)
        print_split()


def set_display_required(continuous: bool):
    import ctypes
    ES_CON = 0x80000000
    ES_DIS = 0x00000002
    if continuous:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_DIS | ES_CON)
    else:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_DIS)


def get_delta():
    if args.show_speed:
        import psutil
        global current_speed
        while start_count_speed:
            before = psutil.net_io_counters().bytes_recv
            time.sleep(0.1)
            now = psutil.net_io_counters().bytes_recv
            delta = (now - before) / (1024 * 102.4)
            current_speed = "  {:.3f} MB/s".format(delta)


def do_download_queue_for_pfs(bucket_path, next_marker=""):
    global has_next
    result_dict = traverse_get_object_keys(bucket_path, next_marker)
    if args.single_file:
        return
    while result_dict["nextmarker"]:
        try:
            result_dict = traverse_get_object_keys(bucket_path, result_dict["nextmarker"])
        except Exception:
            error_map[result_dict["nextmarker"]] = time.time()
            time.sleep(5)
            result_dict = get_object_key(result_dict["nextmarker"])
    if bucket_path == args.bucket_path:
        has_next = False


def traverse_get_object_keys(bucket_path, next_marker=""):
    result_dict = list_pfs_object_keys(bucket_path, next_marker)
    if not result_dict["success"]:
        logging.error(result.text)
        error_map["next_marker" + "@" + next_marker] = result.text
        raise Exception("get pfs object key failed")
    common_pre_fixes = []
    for object_key in result_dict["objectKeys"]:
        object_key_str = object_key["objectKey"]
        size = object_key["size"]
        if args.debug:
            print(f"object key {object_key_str}")
        if object_key_str.endswith("/"):
            if bucket_path == object_key_str or size == "0":
                continue
            common_pre_fixes.append(object_key_str)
        else:
            put_download_queue(object_key)
        if args.single_file:
            break
    traverse_get_object_keys_by_prefix(common_pre_fixes)
    return result_dict


def traverse_get_object_keys_by_prefix(common_pre_fixes):
    for prefix in common_pre_fixes:
        if args.debug:
            print(f"Objects in folder [{prefix}]:")
        do_download_queue_for_pfs(prefix)


def list_pfs_object_keys(bucket_path, next_marker=""):
    path = "/rest/boto3/s3/list/bucket/objectkeys?"
    object_key = base64.urlsafe_b64encode(bucket_path.encode()).decode()
    delimiter = base64.urlsafe_b64encode(args.delimiter.encode()).decode()
    param = f"vendor={args.vendor}&region={args.region}&bucketid={args.bucket_name}" \
            f"&apptoken={args.app_token}&objectkey={object_key}" \
            f"&includeFolder=true&delimiter={delimiter}&nextmarker={next_marker}"
    if args.shared:
        param = param + "&bucketId=" + args.bucket_id
    headers = {
        "Content-Type": "application/json",
        "csb-token": args.app_token
    }
    result = session.get(args.csb_file_server + path + param, headers=headers, timeout=60)
    try:
        return json.loads(result.content)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("list pfs object key error", result.content)
        print("exception is: ", e)
        raise Exception("list pfs object key error")


def put_download_queue(object_key):
    global file_count
    include_flag = True
    exclude_flag = False
    if args.include_file_name is not None or args.exclude_file_name is not None:
        for include_pattern in include_pattern_list:
            # program need to download specific file when include_pattern_list is not none
            # change include_flag to false at first
            include_flag = False
            if include_pattern.search(object_key["objectKey"]) is not None:
                # current object_key need to include
                include_flag = True
                break
        for exclude_pattern in exclude_pattern_list:
            if exclude_pattern.search(object_key["objectKey"]) is not None:
                # current object_key need to exclude
                exclude_flag = True
                break

    if not include_flag or exclude_flag:
        return

    if int(object_key["size"]) > args.big_file:
        big_file_download_queue.put(object_key)
    else:
        download_queue.put(object_key)
        file_count += 1


def get_bucket_type(result_dict):
    bucket_info = result_dict["bucket"]
    bucketType = bucket_info["bucketType"]
    print("bucket type : ", bucketType)
    if bucketType is None:
        print("bucket type is None, use OBJECT")
        return "OBJECT"
    return bucketType


def do_download_queue_by_bucket_type():
    if "PFS" == args.bucket_type.upper():
        do_download_queue_for_pfs(args.bucket_path)
    else:
        do_download_queue()


if __name__ == '__main__':
    check_platform()
    multi_thread_main()
    check_retry()
