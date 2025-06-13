# Encode v1
echo "Test V1"
ASCEND_RT_VISIBLE_DEVICES=4 python ./test_geometry_vae_dump_latent_releasev1.py --config ./configs/exp0417_3D_geometry_vae_releaseV1.py

# echo "Test V2"
# # Encode v2
# ASCEND_RT_VISIBLE_DEVICES=6 python ./test_geometry_vae_dump_latent.py --config ./configs/exp0506_3D_geometry_vae_releaseV2.py
