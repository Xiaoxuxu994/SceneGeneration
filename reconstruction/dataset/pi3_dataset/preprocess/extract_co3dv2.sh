# #!/usr/bin/env bash
# set -euo pipefail

# SRC_DIR="/efs-m3ga/z00533370/reliev3r_data/CO3D-v2CO3D-v2"
# DST_DIR="/efs-m3ga/z00533370/reliev3r_data/CO3Dv2_decode"
# JOBS=8

# mkdir -p "${DST_DIR}"

# echo "[INFO] SRC_DIR=${SRC_DIR}"
# echo "[INFO] DST_DIR=${DST_DIR}"
# echo "[INFO] JOBS=${JOBS}"

# if [[ ! -d "${SRC_DIR}" ]]; then
#   echo "[ERROR] source directory not found: ${SRC_DIR}" >&2
#   exit 1
# fi

# if ! command -v unzip >/dev/null 2>&1; then
#   echo "[ERROR] unzip not found" >&2
#   exit 1
# fi

# export DST_DIR

# find "${SRC_DIR}" -maxdepth 1 -type f -name '*.zip' -print0 \
#   | sort -z \
#   | xargs -0 -n 1 -P "${JOBS}" bash -c '
#       zip_path="$1"
#       zip_name="$(basename "$zip_path")"
#       stem="${zip_name%.zip}"
#       out_dir="${DST_DIR}/${stem}"

#       mkdir -p "$out_dir"

#       echo "[INFO] extracting ${zip_name} -> ${out_dir}"
#       unzip -oq "$zip_path" -d "$out_dir"
#     ' _

# echo "[INFO] extraction done, start chmod..."

# chmod a+rx "${DST_DIR}"
# find "${DST_DIR}" -type d -print0 | xargs -0 -r chmod a+rx
# find "${DST_DIR}" -type f -print0 | xargs -0 -r chmod a+r

# echo "[INFO] done"



#!/usr/bin/env bash
set -euo pipefail

SRC_DIR="/efs-m3ga/z00533370/reliev3r_data/CO3D-v2CO3D-v2"
DST_DIR="/efs-m3ga/z00533370/reliev3r_data/CO3Dv2_decode"
JOBS=8

mkdir -p "${DST_DIR}"

echo "[INFO] SRC_DIR=${SRC_DIR}"
echo "[INFO] DST_DIR=${DST_DIR}"
echo "[INFO] JOBS=${JOBS}"

# if [[ ! -d "${SRC_DIR}" ]]; then
#   echo "[ERROR] source directory not found: ${SRC_DIR}" >&2
#   exit 1
# fi

# if ! command -v unzip >/dev/null 2>&1; then
#   echo "[ERROR] unzip not found" >&2
#   exit 1
# fi

# export DST_DIR

# find "${SRC_DIR}" -maxdepth 1 -type f -name '*.zip' -print0 \
#   | sort -z \
#   | xargs -0 -n 1 -P "${JOBS}" bash -c '
#       set -euo pipefail

#       zip_path="$1"
#       zip_name="$(basename "$zip_path")"
#       stem="${zip_name%.zip}"
#       out_dir="${DST_DIR}/${stem}"
#       done_flag="${out_dir}/.done"
#       fail_flag="${out_dir}/.failed"

#       mkdir -p "$out_dir"

#       if [[ -f "$done_flag" ]]; then
#         echo "[SKIP] ${zip_name} already marked done"
#         exit 0
#       fi

#       # Count files inside zip, excluding directory entries.
#       zip_file_count=$(unzip -Z1 "$zip_path" | grep -v "/$" | wc -l)

#       # Count local extracted files, excluding marker files.
#       local_file_count=$(find "$out_dir" -type f ! -name ".done" ! -name ".failed" | wc -l)

#       if [[ "$zip_file_count" -gt 0 && "$local_file_count" -ge "$zip_file_count" ]]; then
#         touch "$done_flag"
#         rm -f "$fail_flag"
#         echo "[SKIP] ${zip_name} complete: local=${local_file_count}, zip=${zip_file_count}"
#         exit 0
#       fi

#       echo "[RESUME] ${zip_name}: local=${local_file_count}, zip=${zip_file_count}, extracting..."
#       rm -f "$fail_flag"

#       if unzip -oq "$zip_path" -d "$out_dir"; then
#         touch "$done_flag"
#         echo "[DONE] ${zip_name}"
#       else
#         touch "$fail_flag"
#         echo "[FAIL] ${zip_name}" >&2
#         exit 1
#       fi
#     ' _

echo "[INFO] extraction done, start chmod..."

chmod a+rx "${DST_DIR}"
find "${DST_DIR}" -type d -print0 | xargs -0 -r chmod a+rx
find "${DST_DIR}" -type f -print0 | xargs -0 -r chmod a+r

echo "[INFO] done"