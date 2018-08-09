cp "pracovni pro digitalizaci/F_digitalizace krivek/zolerova_g_rhi_hard.txt" .
cp "pracovni pro digitalizaci/F_digitalizace krivek/vodochodska_g_rhi_hard.txt" .
cp "pracovni pro digitalizaci/F_digitalizace krivek/vencelova_g_rhi_hard.txt" .
cp "pracovni pro digitalizaci/F_digitalizace krivek/vanurova_g_rhi_hard.txt" .
cp "pracovni pro digitalizaci/F_digitalizace krivek/trnkova_g_rhi_hard.txt" .
cp "pracovni pro digitalizaci/F_digitalizace krivek/stroblova_g_rhi_hard.txt" .
cp "pracovni pro digitalizaci/F_digitalizace krivek/slavikova_g_rhi_hard.txt" .
cp "pracovni pro digitalizaci/F_digitalizace krivek/samankova_g_rhi_hard.txt" .
cp "pracovni pro digitalizaci/F_digitalizace krivek/prochazkova_g_rhi_hard.txt" .
cp "pracovni pro digitalizaci/F_digitalizace krivek/pallova_g_rhi_hard.txt" .
cp "pracovni pro digitalizaci/F_digitalizace krivek/pacakova_g_rhi_hard.txt" .
cp "pracovni pro digitalizaci/F_digitalizace krivek/ouska_g_rhi_hard.txt" .
cp "pracovni pro digitalizaci/F_digitalizace krivek/ondruskova_g_rhi_hard.txt" .
cp "pracovni pro digitalizaci/F_digitalizace krivek/novakovaMir_g_rhi_hard.txt" .
cp "pracovni pro digitalizaci/F_digitalizace krivek/novakovaKam_g_rhi_hard.txt" .
cp "pracovni pro digitalizaci/F_digitalizace krivek/neumeistrova_g_rhi_hard.txt" .
cp "pracovni pro digitalizaci/F_digitalizace krivek/nemcova_g_rhi_hard.txt" .
cp "pracovni pro digitalizaci/F_digitalizace krivek/majerova_g_rhi_hard.txt" .
cp "pracovni pro digitalizaci/F_digitalizace krivek/kusa_g_rhi_hard.txt" .
cp "pracovni pro digitalizaci/F_digitalizace krivek/kubu_g_rhi_hard.txt" .
cp "pracovni pro digitalizaci/F_digitalizace krivek/kruzliakova_g_rhi_hard.txt" .
cp "pracovni pro digitalizaci/F_digitalizace krivek/kozova_g_rhi_hard.txt" .
cp "pracovni pro digitalizaci/F_digitalizace krivek/knajflova_g_rhi_hard.txt" .
cp "pracovni pro digitalizaci/F_digitalizace krivek/jerabkova_g_rhi_hard.txt" .
cp "pracovni pro digitalizaci/F_digitalizace krivek/hrmova_g_rhi_hard.txt" .
cp "pracovni pro digitalizaci/F_digitalizace krivek/houstkova_g_rhi_hard.txt" .
cp "pracovni pro digitalizaci/F_digitalizace krivek/hosova_g_rhi_hard.txt" .
cp "pracovni pro digitalizaci/F_digitalizace krivek/halkova_g_rhi_hard.txt" .
cp "pracovni pro digitalizaci/F_digitalizace krivek/fliegerova_g_rhi_hard.txt" .
cp "pracovni pro digitalizaci/F_digitalizace krivek/drabova_g_rhi_hard.txt" .
cp "pracovni pro digitalizaci/F_digitalizace krivek/dlouha_g_rhi_hard.txt" .
cp "pracovni pro digitalizaci/F_digitalizace krivek/divnickova_g_rhi_hard.txt" .
cp "pracovni pro digitalizaci/F_digitalizace krivek/cernaMag_g_rhi_hard.txt" .
cp "pracovni pro digitalizaci/F_digitalizace krivek/cernaJan_g_rhi_hard.txt" .
cp "pracovni pro digitalizaci/F_digitalizace krivek/brhlikova_g_rhi_hard.txt" .


grep "g_rhi" "/home/vajicek/Dropbox/krivky_mala/clanek/GRAFY/Vstupni data_ xml a jpg a txt/MF krivky koren nosu hard.xml" |sed -n 's/.*>\(.*\);.*$/cp \"\1\" \./p'|sed 's/\\/\//g'
