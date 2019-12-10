import os
import os.path as pth
import shutil

base_path = 'result'
dst_path = 'arrange_result'

if __name__=='__main__':
    os.makedirs(dst_path, exist_ok=True)

    with open('freemidi_result.tsv', 'w', encoding='utf-8') as f:
        f.write('\t'.join(['File name', 'Artist', 'Song name'])+'\n')
        for artist_name in os.listdir(base_path):
            for song_name in os.listdir(pth.join(base_path, artist_name)):
                file_name = os.listdir(pth.join(base_path, artist_name, song_name))
                if not file_name or file_name[0]=='.mid':
                    continue
                file_name = file_name[0]
                f.write('\t'.join([file_name, artist_name, song_name])+'\n')
                shutil.copy(pth.join(base_path, artist_name, song_name, file_name), dst_path)
