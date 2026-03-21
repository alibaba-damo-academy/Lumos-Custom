#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 wangyongliang <wangyongliang@bogon>
#
# Distributed under terms of the MIT license.

"""

"""
import torch
import requests
import oss2
from io import BytesIO, StringIO
import json
import os
import backoff
from PIL import Image

LOCAL_SCHEME = 'local://'
REMOTE_SCHEME = 'oss://'

BACKOFF_RETRY_ERROR = (
    oss2.exceptions.RequestError,
)


def is_remote(p):
    return p.startswith(REMOTE_SCHEME)


def is_local(p):
    return not is_remote(p)


def parse(p):
    """split schme and path"""

    if is_remote(p):
        return REMOTE_SCHEME, p.split('://')[1]
    else:
        if p.startswith('local://'):
            return LOCAL_SCHEME, p.split('://')[1]
        else:
            return LOCAL_SCHEME, p


class FileSystem(object):
    def __init__(self):
        pass

    def read(self, p, mode):
        raise NotImplementedError()

    def write(self, data, p, mode):
        raise NotImplementedError()


class LocalFileSystem(FileSystem):
    def __init__(self):
        pass

    def read(self, p, mode='rb'):
        scheme, p = parse(p)
        assert scheme == LOCAL_SCHEME, f'expect local scheme, got: {scheme}'
        if mode == 'rb':
            with open(p, 'rb') as f:
                return BytesIO(f.read())
        elif mode == 'r':
            with open(p, 'r') as f:
                return StringIO(f.read())
        else:
            raise ValueError(f"invalid mode: {mode}")

    def read_json(self, p, mode='rb'):
        buf = self.read(p=p, mode=mode)
        return json.loads(buf.getvalue())
    
    def read_pil_image(self, p):
        image = Image.open(p).convert('RGB')
        return image 

    def write(self, data, p, mode='w'):
        scheme, p = parse(p)
        assert scheme == LOCAL_SCHEME, f'expect local scheme, get {scheme}'
        with open(p, mode) as f:
            f.write(data)

    def write_json(self, data, p, mode='w'):
        data = json.dumps(data)
        self.write(data, p, mode=mode)

    def exists(self, p):
        return os.path.exists(p)

    def makedirs(self, p):
        if not self.exists(p):
            os.makedirs(p, exist_ok=True)
        assert os.path.isdir(p), f'{p} is not file.'

    def ls(self, p, recursive=False):
        return os.listdir(p)

    def read_to_local(self, p, lp):
        pass
    
    def resumable_upload(self, localfile, remotefile):
        pass


class RemoteFileSystem(object):
    def __init__(self, cfg_file=None):
        if cfg_file and os.path.exists(cfg_file):
            with open(cfg_file) as f:
                cfg = json.load(f)
        else:
            cfg = {
                'endpoint': os.getenv('OSS_ENDPOINT', ''),
                'accessKeyId': os.getenv('OSS_ACCESS_KEY_ID', ''),
                'accessKeySecret': os.getenv('OSS_ACCESS_KEY_SECRET', '')
            }

        assert cfg.get('endpoint') and cfg.get('accessKeyId') and cfg.get('accessKeySecret'), \
            'Require OSS endpoint, accessKeyId, and accessKeySecret (via config file or environment variables)'

        self.endpoint = cfg['endpoint']
        self.auth = oss2.Auth(cfg['accessKeyId'], cfg['accessKeySecret'])

    def parse(self, p):
        scheme, p = parse(p)
        p = p.split('/')
        bucket = p[0]
        p = '/'.join(p[1:])
        return scheme, bucket, p

    @backoff.on_exception(backoff.expo, BACKOFF_RETRY_ERROR, max_tries=5)
    def ls(self, p, recursive=False):
        '''list all files'''
        scheme, bucket, prefix = self.parse(p)
        assert scheme == REMOTE_SCHEME, f'invalid scheme: {scheme}'
        bucket = oss2.Bucket(self.auth, self.endpoint, bucket)
        filelist = []
        delimiter = '' if recursive else '/'
        for obj in oss2.ObjectIterator(bucket, prefix=prefix, delimiter=delimiter):
            if prefix == obj.key:
                continue
            # if obj.is_prefix():
            # continue
            filelist.append(os.path.relpath(obj.key, prefix))
        return filelist

    @backoff.on_exception(backoff.expo, BACKOFF_RETRY_ERROR, max_tries=5)
    def write_byteio(self, data, p):
        """write bytes to object"""
        _, bucket, p = self.parse(p)
        bucket = oss2.Bucket(self.auth, self.endpoint, bucket)
        bucket.put_object(p, data)

    def write_stringio(self, data, p):
        self.write_byteio(data.encode('utf-8'), p)

    @backoff.on_exception(backoff.expo, BACKOFF_RETRY_ERROR, max_tries=5)
    def append_byteio(self, data, p):
        """append bytes to file"""
        _, bucket, p = self.parse(p)
        bucket = oss2.Bucket(self.auth, self.endpoint, bucket)
        pos = 0
        if bucket.object_exists(p):
            pos = bucket.head_object(p).content_length
        bucket.append_object(p, pos, data)

    def append_stringio(self, data, p):
        self.append_byteio(data.encode('utf-8'), p)

    def write(self, data, p, mode='wb'):
        if mode == 'wb':
            self.write_byteio(data, p)
        elif mode == 'w':
            self.write_stringio(data, p)
        elif mode == 'ab':
            self.append_byteio(data, p)
        elif mode == 'a':
            self.append_stringio(data, p)
        else:
            raise ValueError(f'invalid mode: {mode}')
    
    def resumable_upload(self, localfile, remotefile):
        _, bucket, p = self.parse(remotefile)
        bucket = oss2.Bucket(self.auth, self.endpoint, bucket)
        oss2.resumable_upload(bucket,p,localfile)


    def write_json(self, data, p, mode='rb'):
        buf = json.dumps(data)
        self.write(data=buf, p=p, mode=mode)

    @backoff.on_exception(backoff.expo, BACKOFF_RETRY_ERROR, max_tries=5)
    def read_byteio(self, p):
        """read object to byteio"""
        scheme, bucket, p = self.parse(p)
        bucket = oss2.Bucket(self.auth, self.endpoint, bucket)
        return BytesIO(bucket.get_object(p).read())

    @backoff.on_exception(backoff.expo, BACKOFF_RETRY_ERROR, max_tries=5)
    def read_stringio(self, p, encoding='utf-8'):
        """read stringio"""
        _, bucket, p = self.parse(p)
        bucket = oss2.Bucket(self.auth, self.endpoint, bucket)
        data = bucket.get_object(p).read()
        data = data.decode(encoding='utf-8')
        return StringIO(data)

    def read(self, p, mode='rb'):
        if mode == 'rb':
            return self.read_byteio(p)
        elif mode == 'r':
            return self.read_stringio(p)
        else:
            raise ValueError(f'invalid mode: {mode}')

    def read_json(self, p, mode='rb'):
        buf = self.read(p=p, mode=mode)
        return json.loads(buf.getvalue())
    
    def read_pil_image(self, p):
        buf = self.read(p, mode='rb')
        image = Image.open(buf).convert('RGB')
        return image


    def read_to_local(self, p, lp):
        _, bucket, p = self.parse(p)
        bucket = oss2.Bucket(self.auth, self.endpoint, bucket)
        bucket.get_object_to_file(p, lp)

    @backoff.on_exception(backoff.expo, BACKOFF_RETRY_ERROR, max_tries=5)
    def exists(self, p):
        # oss2 always have a dir
        if p.endswith('/'):
            return True
        _, bucket, p = self.parse(p)
        bucket = oss2.Bucket(self.auth, self.endpoint, bucket)
        return bucket.object_exists(p)

    def makedirs(self, p):
        # any dir is eixsted.
        pass


lfs = LocalFileSystem()
rfs = None
registry = {
    LOCAL_SCHEME: lfs,
}

def init_remote_fs(cfg_file=None):
    """Initialize remote filesystem if needed."""
    global rfs, registry
    if rfs is None:
        rfs = RemoteFileSystem(cfg_file)
        registry[REMOTE_SCHEME] = rfs


def parse_scheme(p):
    if is_local(p):
        return LOCAL_SCHEME
    else:
        return REMOTE_SCHEME


def read(p, mode='rb'):
    scheme = parse_scheme(p)
    if scheme == REMOTE_SCHEME and rfs is None:
        init_remote_fs()
    return registry[scheme].read(p, mode=mode)


def write(data, p, mode='wb'):
    scheme = parse_scheme(p)
    if scheme == REMOTE_SCHEME and rfs is None:
        init_remote_fs()
    return registry[scheme].write(data, p, mode=mode)


def read_json(p, mode='rb'):
    scheme = parse_scheme(p)
    if scheme == REMOTE_SCHEME and rfs is None:
        init_remote_fs()
    return registry[scheme].read_json(p, mode=mode)


def read_pil_image(p):
    scheme = parse_scheme(p)
    if scheme == REMOTE_SCHEME and rfs is None:
        init_remote_fs()
    return registry[scheme].read_pil_image(p)

def read_csv(p, mode='rb', **kwargs):
    import pandas as pd
    bytes_data = read(p, mode=mode)
    metadata = pd.read_csv(bytes_data, **kwargs)
    return metadata



def write_json(data, p, mode='w'):
    scheme = parse_scheme(p)
    if scheme == REMOTE_SCHEME and rfs is None:
        init_remote_fs()
    return registry[scheme].write_json(data, p, mode=mode)


def makedirs(p):
    scheme = parse_scheme(p)
    if scheme == REMOTE_SCHEME and rfs is None:
        init_remote_fs()
    return registry[scheme].makedirs(p)


def exists(p):
    scheme = parse_scheme(p)
    if scheme == REMOTE_SCHEME and rfs is None:
        init_remote_fs()
    return registry[scheme].exists(p)


def read_to_local(p, lp):
    scheme = parse_scheme(p)
    if scheme == REMOTE_SCHEME and rfs is None:
        init_remote_fs()
    return registry[scheme].read_to_local(p, lp)

def resumable_upload(localfile, remotefile):
    scheme = parse_scheme(remotefile)
    if scheme == REMOTE_SCHEME and rfs is None:
        init_remote_fs()
    return registry[scheme].resumable_upload(localfile, remotefile)


def ls(p, recursive=False):
    """List all files in directory."""
    if not p.endswith('/'):
        p += '/'
    scheme = parse_scheme(p)
    if scheme == REMOTE_SCHEME and rfs is None:
        init_remote_fs()
    return registry[scheme].ls(p, recursive)


# torch save and load
def load(f, **kwargs):
    """wrap torch.load to support both local/remote filesystem."""
    return torch.load(read(f, 'rb'), **kwargs)


def save(obj, f, **kwargs):
    """wrap torch.save to support both local/remote filesystem"""
    buf = BytesIO()
    torch.save(obj, buf, **kwargs)
    buf.seek(0)
    write(buf.read(), f, 'wb')