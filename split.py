import os, shutil, random

def split_dataset(base_dir, output_dir, train_ratio=0.75):
    categories = os.listdir(base_dir)
    
    for cat in categories:
        src_dir = os.path.join(base_dir, cat)
        im = os.listdir(src_dir)
        random.shuffle(im)
        
        split_idx = int(len(im) * train_ratio)
        train_im = im[:split_idx]
        test = im[split_idx:]
        
        for sp in ['train', 'test']:
            sp_dir = os.path.join(output_dir, sp, cat)
            os.makedirs(sp_dir, exist_ok=True)
            
        for img in train_im:
            shutil.copy(os.path.join(src_dir, img), os.path.join(output_dir, 'train', cat, img))
        
        for img in test:
            shutil.copy(os.path.join(src_dir, img), os.path.join(output_dir, 'test', cat, img))
            
split_dataset(
    base_dir='C:\\Users\\Hussain Haidary\\OneDrive - LNMIIT\\MyPC\\Documents\\Coding\\Python\\SleepDetection\\dataset_B_Eye_Images',
    output_dir='C:\\Users\\Hussain Haidary\\OneDrive - LNMIIT\\MyPC\\Documents\\Coding\\Python\\SleepDetection\\dataset_B_Eye_Images_split'
)