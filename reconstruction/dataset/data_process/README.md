### Dataset Structure
* Sample from Dataset class:
    ```
    |- images (B,V,3,H,W)
    |- depth_maps (B,V,1,H,W)
    |- camera_extrs (B,V,4,4) w2c
    |- camera_intrs (B,V,3,3) regularized
    |- exhaustive_matchings [(B,K,2) x C(V,2)]
    ```
    