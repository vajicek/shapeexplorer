#!/bin/bash

DATAFOLDER="/home/vajicek/Dropbox/krivky_mala/clanek/GRAFY/Vstupni data_ xml a jpg a txt/"
OUTPUTFOLDER="./projects/malakrivky/data"

extract_filenames () {
  SECTION_NAME=$1
  M_XML_FILE=$2
  OUTPUT_SECTION_NAME=$3
  mkdir -p "${OUTPUTFOLDER}/${OUTPUT_SECTION_NAME}"
  grep "${SECTION_NAME}" "${DATAFOLDER}/${M_XML_FILE}.xml" |
    sed -n "s|.*>\(.*\);.*$|${DATAFOLDER}\1|p"|
    sed 's/\\/\//g'|
    xargs -I{} cp -u "{}" "${OUTPUTFOLDER}/${OUTPUT_SECTION_NAME}"
}

rm -rf ${OUTPUTFOLDER}

extract_filenames "g_rhi" "MF krivky koren nosu hard" "koren_nosu"
extract_filenames "g_rhi" "MF krivky koren nosu soft" "koren_nosu"

extract_filenames "hard nose profile green" "MF krivky hrbet nosu hard zelene body" "hrbet_nosu"
extract_filenames "soft nose profile_CF" "MF krivky hrbet nosu soft CF" "hrbet_nosu"

extract_filenames "ac_mptp" "MF krivky horni ret hard" "horni_ret"
extract_filenames "ac_inc" "MF krivky horni ret soft" "horni_ret"

extract_filenames "inc_gn" "MF krivky dolni ret hard" "dolni_ret"
extract_filenames "inc_gn" "MF krivky dolni ret soft" "dolni_ret"
