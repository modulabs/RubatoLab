import requests
from bs4 import BeautifulSoup
import os
import os.path as pth
import re

import time
import random

import multiprocessing
from tqdm import tqdm


base_url = 'https://freemidi.org'


def get_artist_inform_list_from_genre_url(url):
    res = requests.get(url)
    assert res.status_code != 404, 'Status Code:{}, Url:{}'.format(res.status_code, url)
    artist_list = BeautifulSoup(res.text, 'lxml').find_all('div', class_='genre-link-text')

    def extract_artist_inform_from_div(artist_div):
        artist_name = artist_div.a.get_text().strip()
        artist_url = base_url+'/'+artist_div.a['href']
        return artist_name, artist_url
    artist_inform_list = [extract_artist_inform_from_div(artist_div) for artist_div in artist_list]
    return artist_inform_list


def get_song_inform_list_from_artist_url(url):
    res = requests.get(url)
    assert res.status_code != 404, 'Status Code:{}, Url:{}'.format(res.status_code, url)
    page_num_list = BeautifulSoup(res.text, 'lxml').find('ul', class_='pagination').find_all('li')[1:-1]

    song_div_list = [] 
    for page_num in range(len(page_num_list)):
        each_artist_page_url = artist_url+'-P-'+str(page_num)
        res = requests.get(each_artist_page_url)
        each_song_div_list = BeautifulSoup(res.text, 'lxml').find_all('div', class_='artist-song-cell')
        song_div_list += each_song_div_list

    def extract_song_inform_from_div(song_div):
        song_name = song_div.span.a.get_text().strip()
        song_url = base_url+'/'+song_div.span.a['href']
        return song_name, song_url
    song_inform_list = [extract_song_inform_from_div(song_div) for song_div in song_div_list]
    return song_inform_list


def get_song_detail_from_song_url(url):
    song_id = url.split('/')[-1].split('-')[1]
    song_midi_url = base_url+'/'+'getter'+'-'+song_id
    song_mp3_url = base_url+'/'+'getterm'+'-'+song_id
    return song_id, song_midi_url, song_mp3_url


def download_song_midi_from_url(url, output_path='.'):
    headers = {
        'sec-fetch-mode': "navigate",
        'upgrade-insecure-requests': "1",
        'cache-control': "no-cache",
        'postman-token': "f043755e-0134-9dcd-eaa0-3e003069588f",
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'
    }
    res = requests.get(url, headers=headers, allow_redirects=False)

    try:
        if res.status_code == 302:
            pass
        elif res.status_code == 200:
            return False, 'Url:{}, File is not exist.'
        else:
            raise Exception()

        os.makedirs(output_path, exist_ok=True)
        song_filename = re.findall("filename=(.+)", res.headers['content-disposition'])[0]
        song_fullname = pth.join(output_path, song_filename)
        with open(song_fullname, 'wb') as f:
            f.write(res.content)
    except:
        return False, 'Status Code:{}, Url:{}'.format(res.status_code, url)
    return True, song_fullname


def download_song_mp3_from_url(url, output_path='.'):
    # TODO: Implement this. but we don't need mp3 file now. and it's heavy.
    # It must be implemeted the other library. Because 'requests.get' is slow.
    # To consider multiprocessing
    pass


def download_song_midi_from_url_for_multi(args):
   return download_song_midi_from_url(*args)


def utf8_to_euckr(unicode_string):
    p = re.compile('\xc2|\xa0')
    text = p.sub('', unicode_string)
    text = text.encode('euc-kr', 'replace').decode('euc-kr')
    return text


if __name__=='__main__':
    result_base_path = 'result'

    # ### Single task
    # genre_url = base_url+'/'+'genre-jazz'
    # for artist_name, artist_url in get_artist_inform_list_from_genre_url(genre_url):
    #     for song_name, song_url in get_song_inform_list_from_artist_url(artist_url):
    #         song_id, song_midi_url, song_mp3_url = get_song_detail_from_song_url(song_url)
    #         each_output_path = pth.join(result_base_path, artist_name, song_name)
    #         result = download_song_midi_from_url(song_midi_url, output_path=each_output_path)
    #         print(result)
    #         # time.sleep(random.random()*3)

    ### Multi task
    args_list = []
    genre_url = base_url+'/'+'genre-jazz'
    for artist_name, artist_url in tqdm(get_artist_inform_list_from_genre_url(genre_url)):
        for song_name, song_url in get_song_inform_list_from_artist_url(artist_url):
            song_id, song_midi_url, song_mp3_url = get_song_detail_from_song_url(song_url)
            each_output_path = pth.join(result_base_path, artist_name, song_name)
            args_list.append([song_midi_url, each_output_path])

    success_cnt = 0
    pool = multiprocessing.Pool(processes=16)
    for is_success, result_text in tqdm(pool.imap_unordered(download_song_midi_from_url_for_multi, args_list), total=len(args_list)):
        if is_success:
            success_cnt += 1
            print('Success!', utf8_to_euckr(result_text))
        else:
            print('Failed!', utf8_to_euckr(result_text))

    print()
    print('Total:{}, Success:{}, Failed:{}'.format(len(args_list), success_cnt, len(args_list)-success_cnt))