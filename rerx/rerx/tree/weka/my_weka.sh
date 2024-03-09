# ai_lab_1dcnn_project/codes/rerx/rerx_origin

main() {
  weka_dir=$1
  nominal_cols=$2
  datas_arff=$3
  result_txt=$4
  mode=$5
  tree=$6
  min_instance=$7
  pruning_conf=$8
  data_path=$9

  # change csv to arff
  if [ "$mode" = "cont" ] ; then
    java -cp $weka_dir/weka.jar weka.core.converters.CSVLoader $data_path -N "last" -B 10000 > $datas_arff
  elif [ "$mode" = "ori" ] && [ "$nominal_cols" != "\"\"" ] ; then
    java -cp $weka_dir/weka.jar weka.core.converters.CSVLoader $data_path -N "${nominal_cols},last" -B 10000 > $datas_arff
  elif [ "$mode" = "ori" ] && [ "$nominal_cols" = "\"\"" ] ; then
    java -cp $weka_dir/weka.jar weka.core.converters.CSVLoader $data_path -N "last" -B 10000 > $datas_arff
  elif [ -z "$nominal_cols" ] ; then
    echo "\n" \
         "=======================================================\n" \
         "Error: No nominal value in dataset. Choose mode [cont].\n" \
         "=======================================================\n\n"
    exit 1
  else
    echo "\n" \
         "=================================\n" \
         "Error: mode selection -> $mode\n" \
         "=================================\n\n"
    exit 1
  fi

  # fit tree
  if [ "$tree" = "j48" ] ; then
    java -cp $weka_dir/weka.jar weka.classifiers.trees.J48 \
      -t $datas_arff \
      | tee $result_txt > /dev/null
      #> tmp_dt_train.txt
  elif [ "$tree" = "j48graft" ] ; then
    java -cp $weka_dir/old_weka.jar weka.classifiers.trees.J48graft \
      -M $min_instance \
      -C $pruning_conf \
      -t $datas_arff \
      | tee $result_txt > /dev/null
  else
    echo "\n" \
         "==================================\n" \
         "Error: tree selection -> $tree\n"
         "==================================\n\n"
    exit 1
  fi
}

if [ $# -eq 9 ] ; then
  # $1 ... weka path
  # $2 ... nominal colmns
  # $3 ... output arff path
  # $4 ... rules path made by weka
  # $5 ... mode: ori or cont
  # $6 ... tree: j48 or j48graft
  # $7 ... min instance of j48graft
  # $8 ... pruning threshold
  # $9 ... csv path created by .py
  # echo $1 $2 $3 $4 $5 $6 $7 $8 $9
  main $1 $2 $3 $4 $5 $6 $7 $8 $9
else
  exit 1
fi
