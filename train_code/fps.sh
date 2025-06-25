# nohup python gen_fps.py --npuid 1 --startid 0 --endid 150 --path ../../data/scannetpp_pcd/train --outpath ../../data/scannetpp_pcd/train_fps > fps1.txt 2>&1 &
# nohup python gen_fps.py --npuid 2 --startid 150 --endid 300 --path ../../data/scannetpp_pcd/train --outpath ../../data/scannetpp_pcd/train_fps > fps2.txt 2>&1 &
# nohup python gen_fps.py --npuid 3 --startid 300 --endid 450 --path ../../data/scannetpp_pcd/train --outpath ../../data/scannetpp_pcd/train_fps > fps3.txt 2>&1 &
# nohup python gen_fps.py --npuid 5 --startid 450 --endid 600 --path ../../data/scannetpp_pcd/train --outpath ../../data/scannetpp_pcd/train_fps > fps4.txt 2>&1 &
# nohup python gen_fps.py --npuid 6 --startid 600 --endid 750 --path ../../data/scannetpp_pcd/train --outpath ../../data/scannetpp_pcd/train_fps > fps5.txt 2>&1 &
# nohup python gen_fps.py --npuid 7 --startid 750 --endid 856 --path ../../data/scannetpp_pcd/train --outpath ../../data/scannetpp_pcd/train_fps > fps6.txt 2>&1 &
# nohup python gen_fps.py --npuid 1 --startid 0 --endid 50 --path ../../data/scannetpp_pcd/val --outpath ../../data/scannetpp_pcd/val_fps > fps7.txt 2>&1 &
nohup python gen_fps_v2.py --npuid 4 --startid 150 --endid 856 --path ../../data/scannetpp_pcd/train --outpath ../../data/scannetpp_pcd/train_fps_fix > fps1.txt 2>&1 &
