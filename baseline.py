def calculate_f1(confusion):
    true_positive = confusion[0][0]
    false_postitive = confusion[1][0]
    false_negative = confusion[0][1]
    precision = true_positive/(true_positive+false_postitive)
    recall = true_positive/(true_positive+false_negative)
    return 2 * (precision + recall)/(precision * recall)  # f1


def baseline(np_data, np_label):
    '''
    Retreive the baseline stats, by calculating the mean of the TD
    group for MLU words, NDW100, and Total Utterances. Then comparing
    whether the samples are 1.25 SD below the mean.
    '''
    # Confusion matrices
    baseline_confusion = [[0, 0], [0, 0]]

    # Get the features from the data
    MLU_words = np_data[27]
    NDW = np_data[11]
    total_utts = np_data[25]

    # Isolate the TD samples
    deletion_array = []
    for i in range(0, len(MLU_words)):
        if np_label[i] == 1:  # Is SLI
            deletion_array.append(i)

    # Create a new array with only TD
    td_mlu = np.delete(MLU_words, deletion_array)
    td_ndw = np.delete(NDW, deletion_array)
    td_tu = np.delete(total_utts, deletion_array)

    # Calculate the cutoff
    mlu_cutoff = np.std(td_mlu) - np.mean(td_mlu)
    ndw_cutoff = np.std(td_ndw) - np.mean(td_ndw)
    total_utts_cuttoff = np.std(td_tu) - np.mean(td_tu)

    # Create the confusion matricies
    for i in range(0, len(MLU_words)):
        sample_mlu = MLU_words[i]
        sample_ndw = NDW[i]
        sample_tu = total_utts[i]
        true_label = np_label[i]
        if (sample_mlu < mlu_cutoff and sample_ndw < ndw_cutoff) or \
            (sample_mlu < mlu_cutoff and sample_tu < total_utts_cuttoff) or \
                (sample_tu < total_utts_cuttoff and sample_ndw < ndw_cutoff):
            sli = 1
            if sli == true_label:
