"""
python 3.8

"""

from skmultilearn.dataset import load_from_arff
from skmultilearn.dataset import save_to_arff
import os


dt = "GNSyn-dim-1001-ins-20000-skew1-concept-drift-2000-3000.arff"
file_path = "F:/PhD/#multi-label classification/datasets/meka experimentaion/"+dt
output_file_name = "F:/PhD/#multi-label classification/datasets/meka experimentaion/Synthectiv-Sparse.arff"

label_count = 25
label_location="start"
arff_file_is_sparse = False


file_output_path = file_path+"-temp"
f_ouput_stream = open(file_output_path,'w')
data_started = False
with open(file_path, 'r') as input_file_stream:
    for cnt, line in enumerate(input_file_stream):
        if data_started:
            if line.strip()[-1] == ",":
                line=line.strip()[0:-1]
                line= (line+"\n")
        try:
            if len(line.strip()) > 0 and line.strip()[0] == "@" :
                if line.strip().lower() == "@data":
                    data_started = True
        except:
            print("Line number:", cnt)
        f_ouput_stream.write(line)


f_ouput_stream.close()
os.remove(file_path)
os.rename(file_output_path, file_path)

X, y, feature_names, label_names = load_from_arff(file_path,label_count=label_count,label_location=label_location,load_sparse=arff_file_is_sparse,return_attribute_definitions=True)

save_to_arff(X, y, label_location='end', save_sparse=True, filename=output_file_name)


