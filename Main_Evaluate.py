

from ssclassifier import ccutils as ut

original_file = "moa experimentaion/20NG-d1006.arff"
# original_file = "moa experimentaion/30-Yahoo_Business-d21920.arff"

# pred_file_name= "../MLSAMPkNN/30-Yahoo_Business-d21920.pred"
pred_file_name= "../#results/AMRules/20NG-AMR.pred"


headers, common_string = ut.evaluate_moa_prediction_file(source_arff_file=original_file, moa_prediction_file=pred_file_name)
print(headers)
print(common_string)