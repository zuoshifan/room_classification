import os
import os.path as path
import glob
# import shutil
import scipy.misc as misc


src_dir = './downloads/'
dest_dir = './data/'

train_size = 1600
validation_size = 320
img_width, img_height = 150, 150

keywords = ['bathroom', 'bedroom', 'livingroom', 'kitchen']

for keyword in keywords:
    files = glob.glob(src_dir+'%s/*.jpg' % keyword)
    # print len(files)
    # print path.basename(files[0]), path.basename(files[0]).split('.')[0]
    # print path.basename(files[1]), path.basename(files[1]).split('.')[0]
    
    for fi, fl in enumerate(files):
        fl_name = path.basename(fl)
        idx, suffix = fl_name.split('.')
        if fi < train_size:
            new_name = 'train/%s/%s%04d.%s' % (keyword, keyword, fi+1, suffix)
        elif fi < train_size + validation_size:
            new_name = 'validation/%s/%s%04d.%s' % (keyword, keyword, fi-train_size+1, suffix)
        else:
            break
        new_fl_name = dest_dir+new_name
        new_fl_path = path.dirname(new_fl_name)
        if not path.isdir(new_fl_path):
            os.makedirs(new_fl_path)
    
        # copy file to new dir
        # shutil.copyfile(fl, new_fl_name)
    
        # resize image and save to file
        img = misc.imread(fl)
        new_img = misc.imresize(img, [img_width, img_height], interp='nearest')
        misc.imsave(new_fl_name, new_img)
