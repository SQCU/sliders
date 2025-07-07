# REFERENCE: RETROGRADE SAMPLING
scales_to_look = random.sample(list(scales_unique),2)   #2 choices
scales_to_look.sort()   #smaller first idx

#use lowest then highest
folder1 = folders[scales==scales_to_look[-1]][0]
folder2 = folders[scales==scales_to_look[0]][0]

ims = os.listdir(f'{folder_main}/{folder1}/')
ims = [im_ for im_ in ims if '.png' in im_ or '.jpg' in im_ or '.jpeg' in im_ or '.webp' in im_]
random_sampler = random.randint(0, len(ims)-1)

#...
img1 = Image.open(f'{folder_main}/{folder1}/{ims[random_sampler]}').resize((512,512))#
img2 = Image.open(f'{folder_main}/{folder2}/{ims[random_sampler]}').resize((512,512))#