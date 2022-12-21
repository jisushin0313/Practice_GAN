pip3 install pytorch-fid

touch evaluation.tsv

cd submission
for FILE in *.zip; do
  unzip $FILE;
  echo -n "${FILE::(-4)}" > ../evaluation.tsv
  echo -ne "\t" >> ../evaluation.tsv
  python -m pytorch_fid "${FILE::(-4)}" ../real >> ../evaluation.tsv
done
cd ../

