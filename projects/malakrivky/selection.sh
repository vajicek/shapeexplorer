#!/bin/bash

DATAFOLDER="/home/vajicek/Dropbox/krivky_mala/clanek/GRAFY/Vstupni data_ xml a jpg a txt/"

extract_filenames () {
  SECTION_NAME=$1
  M_XML_FILE=$2
  mkdir -p data/${SECTION_NAME}
  grep ${SECTION_NAME} "${DATAFOLDER}/${M_XML_FILE}.xml" |
    sed -n "s|.*>\(.*\);.*$|${DATAFOLDER}\1|p"|
    sed 's/\\/\//g'|
    xargs -I{} cp -u "{}" data/${SECTION_NAME}
}


extract_filenames "g_rhi" "MF krivky koren nosu hard"
extract_filenames "g_rhi" "MF krivky koren nosu soft"

#extract_filenames "g_rhi" "MF krivky hrbet nosu hard zelene body"
#extract_filenames "g_rhi" "MF krivky hrbet nosu soft zelene body muzi cervene zeny"

extract_filenames "ac_mptp" "MF krivky horni ret hard"
extract_filenames "ac_inc" "MF krivky horni ret soft"

extract_filenames "inc_gn" "MF krivky dolni ret hard"
extract_filenames "inc_gn" "MF krivky dolni ret soft"
