nohup python gen_fps_datasetv3.py --npuid 0 --startid 0 --endid 50 --path /mnt/data1/z00912387/scannetpp_pcd/val --outpath ../../data/scannetpp_pcd/val_fps_v3 > val_fps.log 2>&1 &
nohup python gen_fps_datasetv4.py --npuid 1 --startid 0 --endid 856 --path /mnt/data1/z00912387/scannetpp_pcd/train --outpath ../../data/scannetpp_pcd/train_fps_v4 > train_fps.txt 2>&1 &
