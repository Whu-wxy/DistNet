import os
import json

def convert(src_path, dst_path, ignore_english=False):
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)

    sources = os.listdir(src_path)
    for src in sources:
        path = os.path.join(src_path, src)
        with open(path, encoding='utf-8', mode='r') as f:
            json_data = json.load(f)
            char_list = json_data['chars']
            with open(os.path.join(dst_path, src.split('.')[0]+'.txt'), encoding='utf-8', mode='w') as f2:
                for char in char_list:
                    if ignore_english:
                        if (u'\u0041' <= str(char['transcription']) <= u'\u005a') or \
                                (u'\u0061' <= str(char['transcription']) <= u'\u007a'):
                            continue

                    pts = list(map(int, char['points']))
                    f2.write(', '.join(('%s' %pt for pt in pts)))

                    # if char['ignore'] == 0:
                    #     f2.write(', '+str(char['transcription']))
                    # else:
                    #     f2.write(', '+"###")
                    f2.write('\n')

if __name__ == '__main__':
    convert('./gt_json', './gt', True)
    print('finished.')
