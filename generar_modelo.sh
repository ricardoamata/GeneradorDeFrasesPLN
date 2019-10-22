#!/bin/bash
# you need to export the variable KENLM_PATH in order to this to work
gen_arp="${KENLM_PATH}/build/bin/lmplz" 
gen_bin="${KENLM_PATH}/build/bin/build_binary"

$gen_arp -o 5 --text datos/texto_entrenamiento.txt --arpa quijote_model.arpa
$gen_bin -s quijote_model.arpa quijote_lm.binary
