My attempt at recreating the feature selection in Gabani, doesn't quite work

def isolate_features(X_data):
Removes some features from the analysis
# GROUP 1: Language productivity
group1_only = X_data[0:6]
group1_excluded = X_data[6:]

# GROUP 2: Morphosyntactic skills
group2_only = X_data[6:11]
group2_excluded = X_data[0:6] + X_data[11:]

# GROUP 3: Vocabulary Knowledge
group3_only = []
group3_only.append(X_data[11])
group3_excluded = X_data[0:11] + X_data[12:]

# Group 4: Speech fluency
group4_only = X_data[12:15]
group4_excluded = X_data[0:12] + X_data[15:]

# GROUP 5: LM probabilites
group5_only = X_data[15:23]
group5_excluded = X_data[0:15] + X_data[23:]

# GROUP 6: Standard Scores
group6_only = X_data[23:31]
group6_excluded = X_data[0:23] + X_data[31:]

# GROUP 7: Sentence Complexity
group7_only = X_data[31:58]
group7_excluded = X_data[0:31] + X_data[58:]

# GROUP 8: Error rates
group8_only = X_data[58:]
group8_excluded = X_data[0:58]

# For feature selection
only_conti4_fs, excluded_conti4_fs = isolate_features(X_data[0])
only_enni_fs, excluded_enni_fs = isolate_features(X_data[1])
only_gillam_fs, excluded_gillam_fs = isolate_features(X_data[2])

corpus_only = [only_conti4_fs, only_enni_fs, only_gillam_fs]
corpus_excluded = [excluded_conti4_fs, excluded_enni_fs,
excluded_gillam_fs]
only_features = []
excluded_features = []
# For the feature extraction
for j in range(0, len(only_gillam_fs)):
only_np_data, only_np_label = \
    numpify_arrays(corpus_only[i][j], Y_data[i])
excluded_np_data, excluded_np_label = \
    numpify_arrays(corpus_excluded[i][j], Y_data[i])
only_features.append((only_np_data, only_np_label))
excluded_features.append((excluded_np_data, excluded_np_label))

feature_X.append((only_features, excluded_features))
only_groups = [group1_only, group2_only, group3_only, group4_only,
group5_only, group6_only, group7_only, group8_only]
excluded_groups = [group1_excluded, group2_excluded, group3_excluded,
group4_excluded, group5_excluded, group6_excluded,
group7_excluded, group8_excluded]

return only_groups, excluded_groups

    # Now run the feature selection
    group_labels = ['Conti4_O1', 'Conti4_02', 'Conti4_03', 'Conti4_04',
    'Conti4_05', 'Conti4_06', 'Conti4_07', 'Conti4_08',
    'Conti4_E1', 'Conti4_E2', 'Conti4_E3', 'Conti4_E3',
    'Conti4_E4', 'Conti4_E5', 'Conti4_E6', 'Conti4_E7',
    'Conti4_E8', 'ENNI_O1', 'ENNI_O2', 'ENNI_O3',
    'ENNI_O4', 'ENNI_O5', 'ENNI_O6', 'ENNI_O7',
    'ENNI_O8', 'ENNI_P1', 'ENNI_P2', 'ENNI_P3',
    'ENNI_P4', 'ENNI_P5', 'ENNI_P6', 'ENNI_P7',
    'ENNI_P8', 'Gillam_O1', 'Gillam_O2', 'Gillam_O3',
    'Gillam_O4', 'Gillam_O5', 'Gillam_O6', 'Gillam_O7',
    'Gillam_O8', 'Gillam_P1', 'Gillam_P2', 'Gillam_P3',
    'Gillam_P4', 'Gillam_P5', 'Gillam_P6', 'Gillam_P7',
    'Gillam_P8']
    gl_index = 0  # Group label index
    for corpus in feature_X:
    for incl_excl in corpus:
    for data, label in incl_excl:
    tag = [group_labels[gl_index], 'hi']
    featurewriter.writerow(tag)
    save_png = group_labels[gl_index] + '.png'
    scores = run_classifiers(data, label, save_png)
    scores.to_csv(fcsvfile, mode='a')
    gl_index += 1
'''
