import glob


print 'train bedroom:', len(glob.glob('./data/train/bedroom/*.jpg'))
print 'train bathroom:', len(glob.glob('./data/train/bathroom/*.jpg'))
print 'validation bedroom:', len(glob.glob('./data/validation/bedroom/*.jpg'))
print 'validation bathroom:', len(glob.glob('./data/validation/bathroom/*.jpg'))

