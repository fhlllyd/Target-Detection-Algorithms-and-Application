# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Утилиты для скачивания
"""

import logging
import os
import subprocess
import urllib
from pathlib import Path

import requests
import torch


def is_url(url, check=True):
    # Проверяет, является ли строка URL, и проверяет, существует ли URL
    try:
        url = str(url)
        result = urllib.parse.urlparse(url)
        assert all([result.scheme, result.netloc])  # проверка, является ли строкой URL
        return (urllib.request.urlopen(url).getcode() == 200) if check else True  # проверка, существует ли онлайн
    except (AssertionError, urllib.request.HTTPError):
        return False


def gsutil_getsize(url=''):
    # Получает размер файла в байтах с помощью gsutil du для URL Google Cloud Storage
    s = subprocess.check_output(f'gsutil du {url}', shell=True).decode('utf-8')
    return eval(s.split(' ')[0]) if len(s) else 0  # возвращает размер в байтах


def url_getsize(url='https://ultralytics.com/images/bus.jpg'):
    # Возвращает размер скачиваемого файла в байтах
    response = requests.head(url, allow_redirects=True)
    return int(response.headers.get('content-length', -1))


def safe_download(file, url, url2=None, min_bytes=1E0, error_msg=''):
    # Пытается скачать файл из url или url2, проверяет и удаляет незавершенные скачивания < min_bytes
    from utils.general import LOGGER

    file = Path(file)
    assert_msg = f"Скачанный файл '{file}' отсутствует или размер меньше min_bytes={min_bytes}"
    try:  # url1
        LOGGER.info(f'Скачивание {url} в {file}...')
        torch.hub.download_url_to_file(url, str(file), progress=LOGGER.level <= logging.INFO)
        assert file.exists() and file.stat().st_size > min_bytes, assert_msg  # проверка
    except Exception as e:  # url2
        if file.exists():
            file.unlink()  # удаляет частичные скачивания
        LOGGER.info(f'ОШИБКА: {e}\nПовторная попытка скачивания {url2 or url} в {file}...')
        os.system(f"curl -# -L '{url2 or url}' -o '{file}' --retry 3 -C -")  # скачивание с помощью curl, повторяет при ошибке и возобновляет скачивание
    finally:
        if not file.exists() or file.stat().st_size < min_bytes:  # проверка
            if file.exists():
                file.unlink()  # удаляет частичные скачивания
            LOGGER.info(f"ОШИБКА: {assert_msg}\n{error_msg}")
        LOGGER.info('')


def attempt_download(file, repo='ultralytics/yolov5', release='v7.0'):
    # Пытается скачать файл с GitHub release assets, если он не найден локально. release = 'latest', 'v7.0', и т.д.
    from utils.general import LOGGER

    def github_assets(repository, version='latest'):
        # Возвращает тег GitHub repo (например, 'v7.0') и assets (например, ['yolov5s.pt', 'yolov5m.pt', ...])
        if version != 'latest':
            version = f'tags/{version}'  # например, tags/v7.0
        response = requests.get(f'https://api.github.com/repos/{repository}/releases/{version}').json()  # github api
        return response['tag_name'], [x['name'] for x in response['assets']]  # тег, assets

    file = Path(str(file).strip().replace("'", ''))
    if not file.exists():
        # URL указан
        name = Path(urllib.parse.unquote(str(file))).name  # декодирует '%2F' в '/' и т.д.
        if str(file).startswith(('http:/', 'https:/')):  # скачивание
            url = str(file).replace(':/', '://')  # Pathlib превращает :// в :/
            file = name.split('?')[0]  # парсинг аутентификации https://url.com/file.txt?auth...
            if Path(file).is_file():
                LOGGER.info(f'Найдено {url} локально в {file}')  # файл уже существует
            else:
                safe_download(file=file, url=url, min_bytes=1E5)
            return file

        # GitHub assets
        assets = [f'yolov5{size}{suffix}.pt' for size in 'nsmlx' for suffix in ('', '6', '-cls', '-seg')]  # по умолчанию
        try:
            tag, assets = github_assets(repo, release)
        except Exception:
            try:
                tag, assets = github_assets(repo)  # последний release
            except Exception:
                try:
                    tag = subprocess.check_output('git tag', shell=True, stderr=subprocess.STDOUT).decode().split()[-1]
                except Exception:
                    tag = release

        file.parent.mkdir(parents=True, exist_ok=True)  # создает родительскую директорию (если требуется)
        if name in assets:
            url3 = 'https://drive.google.com/drive/folders/1EFQTEUeXWSFww0luse2jB9M1QNZQGwNl'  # зеркало на Google Drive
            safe_download(
                file,
                url=f'https://github.com/{repo}/releases/download/{tag}/{name}',
                min_bytes=1E5,
                error_msg=f'{file} отсутствует, попробуйте скачать с https://github.com/{repo}/releases/{tag} или {url3}')

    return str(file)