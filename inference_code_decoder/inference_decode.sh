# Decode v1
echo "Decode V1"
ASCEND_RT_VISIBLE_DEVICES=3 python ./test_geometry_vae_decode_latent_releasev1.py --config ./configs/exp0417_3D_geometry_vae_releaseV1.py

# Decode v2
echo "Decode V2"
ASCEND_RT_VISIBLE_DEVICES=3 python ./test_geometry_vae_decode_latent.py --config ./configs/exp0506_3D_geometry_vae_releaseV2.py