from pathlib import Path
from tqdm import tqdm 

if __name__ == '__main__':
    import sys, os
    split_dir = Path(sys.argv[1])
    path_lowres = Path(sys.argv[2])
    path_highres = Path(sys.argv[3])
    
    all_used = list((Path(split_dir) / "train.txt").read_text().splitlines()) + list((Path(split_dir) / "val.txt").read_text().splitlines()) + list((Path(split_dir) / "test.txt").read_text().splitlines())
    all_on_disk = list([x.name.split('.')[0] for x in path_highres.iterdir()])
    all_unused = list(set(all_on_disk) - set(all_used))
    print(f"Used/Unused: {len(all_unused)}/{len(all_on_disk)}, Used: {len(all_on_disk) - len(all_unused)} = {len(all_used)}")
    for i in tqdm(all_unused):
        lr_path = path_lowres / (i + ".npy")
        hr_path = path_highres / (i + ".npy")
        os.remove(lr_path)        
        os.remove(hr_path)
