#!/bin/bash
eos_cmd="eos root://cmseos.fnal.gov"
DRY_RUN=${DRY_RUN:-0}

run_cmd() {
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[DRY] $*"
    else
        "$@"
    fi
}

############################
# BASE PATHS
############################
source_signal_base="/eos/uscms/store/group/lpcml/bbbam/signals_h5_from_miniAOD_July_2026"
source_bkg_base="/eos/uscms/store/group/lpcml/bbbam/backgrounds_h5_from_miniAOD_July_2026"

sig_train_base="/eos/uscms/store/group/lpcml/bbbam/signal_h5_from_miniAOD_train_July_2026"
sig_valid_base="/eos/uscms/store/group/lpcml/bbbam/signal_h5_from_miniAOD_valid_July_2026"

bkg_train_base="/eos/uscms/store/group/lpcml/bbbam/background_h5_from_miniAOD_train_July_2026"
bkg_valid_base="/eos/uscms/store/group/lpcml/bbbam/background_h5_from_miniAOD_valid_July_2026"

############################
# CREATE MAIN DIRS
############################
run_cmd $eos_cmd mkdir -p "$sig_train_base"
run_cmd $eos_cmd mkdir -p "$sig_valid_base"
run_cmd $eos_cmd mkdir -p "$bkg_train_base"
run_cmd $eos_cmd mkdir -p "$bkg_valid_base"
echo "Directories created."

############################
# SPLIT TABLE
############################
# -------- SIGNALS --------
declare -A S_TRAIN
declare -A S_VALID
S_TRAIN["3p7"]=65;  S_VALID["3p7"]=3
S_TRAIN["4"]=66;    S_VALID["4"]=4
S_TRAIN["5"]=68;    S_VALID["5"]=4
S_TRAIN["6"]=70;    S_VALID["6"]=4
S_TRAIN["8"]=73;    S_VALID["8"]=4

# -------- BACKGROUNDS --------
declare -A B_TRAIN
declare -A B_VALID
B_TRAIN["QCD"]=102;      B_VALID["QCD"]=6
B_TRAIN["Wjets"]=791;    B_VALID["Wjets"]=48
B_TRAIN["DYto2L"]=1356;  B_VALID["DYto2L"]=83
B_TRAIN["TTbar"]=141;    B_VALID["TTbar"]=8
B_TRAIN["HTo2Tau"]=235;  B_VALID["HTo2Tau"]=14

############################
# FUNCTION: SPLIT AND MOVE
############################
move_files_split () {
    src_dir=$1
    train_dir=$2
    valid_dir=$3
    n_train=$4
    n_valid=$5
    label=$6

    # Check source exists
    if ! $eos_cmd ls "$src_dir" &>/dev/null; then
        echo "ERROR: Source not found: $src_dir"
        return 1
    fi

    run_cmd $eos_cmd mkdir -p "$train_dir"
    run_cmd $eos_cmd mkdir -p "$valid_dir"

    all_files=$($eos_cmd ls "$src_dir" | grep "\.h5$" | sort)
    total=$(echo "$all_files" | wc -l)

    if [[ $total -lt $((n_train + n_valid)) ]]; then
        echo "WARNING: $label — only $total files available, need $((n_train + n_valid))"
    fi

    train_files=$(echo "$all_files" | head -n "$n_train")
    valid_files=$(echo "$all_files" | tail -n +$((n_train + 1)) | head -n "$n_valid")

    train_count=0
    for f in $train_files; do
        echo "  [train] $label : $f"
        if run_cmd $eos_cmd mv "$src_dir/$f" "$train_dir/$f"; then
            ((train_count++))
        else
            echo "  FAILED train: $f"
        fi
    done

    valid_count=0
    for f in $valid_files; do
        echo "  [valid] $label : $f"
        if run_cmd $eos_cmd mv "$src_dir/$f" "$valid_dir/$f"; then
            ((valid_count++))
        else
            echo "  FAILED valid: $f"
        fi
    done

    echo "DONE $label -> train: $train_count, valid: $valid_count"
}

############################
# PROCESS SIGNALS
############################
echo "==== SIGNALS ===="
for mass in "${!S_TRAIN[@]}"; do
    subdir="IMG_signal_mass_${mass}_GeV_miniAOD"
    src="${source_signal_base}/${subdir}"
    train_dir="${sig_train_base}/${subdir}"
    valid_dir="${sig_valid_base}/${subdir}"
    move_files_split "$src" "$train_dir" "$valid_dir" \
        "${S_TRAIN[$mass]}" "${S_VALID[$mass]}" "signal_${mass}"
done

############################
# PROCESS BACKGROUNDS
############################
echo "==== BACKGROUNDS ===="
for bkg in "${!B_TRAIN[@]}"; do
    subdir="IMG_background_${bkg}_miniAOD"
    src="${source_bkg_base}/${subdir}"
    train_dir="${bkg_train_base}/${subdir}"
    valid_dir="${bkg_valid_base}/${subdir}"
    move_files_split "$src" "$train_dir" "$valid_dir" \
        "${B_TRAIN[$bkg]}" "${B_VALID[$bkg]}" "bkg_${bkg}"
done

echo "ALL DONE ✔"
