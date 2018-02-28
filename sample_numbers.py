import glob


types = ['train', 'validation']
keywords = ['bathroom', 'bedroom', 'livingroom', 'kitchen']

for tp in types:
    for kw in keywords:
        print '%s %s:' % (tp, kw), len(glob.glob('./data/%s/%s/*.jpg' % (tp, kw)))
