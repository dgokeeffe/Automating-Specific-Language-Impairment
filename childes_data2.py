# -*- coding: utf-8 -*-
"""

childes_data
~~~~~~~~~~~~~~~

Creates a class that holds all the CHILDES data from the ENNI and the
Gillam corpora and then generates each of the features

"""

import nltk
from nltk.corpus.reader import CHILDESCorpusReader
from nltk.corpus import brown
from nltk.tag import tnt
# from random import shuffle
import pandas as pd
import numpy as np
import json
import os
import itertools
import string


# =============================================================================
# Class CHILDESdata
# NOTE: X data is of form [[features], [features], ...,]
#       Y data is of form [SLI, TD, SLI, TD, SLI, ...,]
# =============================================================================


class CHILDESdata:
    def __init__(self, preload=False):
        if preload is True:
            try:
                x_data = open("x.txt", "r")
                y_data = open("y.txt", "r")
            except IOError:
                print('Could not open x or y data file')
            self.X = json.load(x_data)
            self.Y = json.load(y_data)
            self.list_of_corpora = ['Conti4', 'ENNI', 'Gillam', 'EG']
            self.len_of_corpora = [19, 99, 77, 300, 171, 497]  # Hard coded
        else:
            self.X = []  # Where the features will go as list of lists
            self.Y = []  # Index corresponds to the label (SLI or TD)
            # Create the corpus readers for each corpus
            nltk.data.path.append(os.getcwd() + '/')
            # Add the CWD to the datapath
            self.corpus_root = nltk.data.find('Data/')
            sli_conti4 = CHILDESCorpusReader(self.corpus_root,
                                             'SLI/Conti4/.*.xml')
            td_conti4 = CHILDESCorpusReader(self.corpus_root,
                                            'TD/Conti4/.*.xml')
            sli_enni = CHILDESCorpusReader(self.corpus_root,
                                           'SLI/ENNI/.*.xml')
            td_enni = CHILDESCorpusReader(self.corpus_root,
                                          'TD/ENNI/.*.xml')
            sli_gil = CHILDESCorpusReader(self.corpus_root,
                                          'SLI/Gillam/.*.xml')
            td_gil = CHILDESCorpusReader(self.corpus_root,
                                         'TD/Gillam/.*.xml')
            # Corpora MUST BE of form [C1 SLI, C1 TD, C2 SLI, ..., Cn TD]
            self.corpora = [sli_conti4, td_conti4, sli_enni, td_enni, sli_gil,
                            td_gil]
            self.list_of_corpora = ['Conti4', 'ENNI', 'Gillam']
            self.len_of_corpora = []
            for corpus in self.corpora:
                self.len_of_corpora.append(len(corpus.fileids()))
            # Extract the data needed from the corpora only get the child data
            self.words, self.corpus_ids = self.get_words_and_ids()
            self.sents = self.get_sents()
            try:
                pos_sents_tnt = open("tagged_data.txt", "r")
            except IOError:
                print('Could not open tagged data file')
            self.tnt_sents_tagged = json.load(pos_sents_tnt)
            self.Y = self.make_y()

    def get_X_data(self, corpus='all3'):
        '''
        Returns the individual corpus data
        Arguments:
            corpus - A string naming the corpus to extract
                     (Conti4, ENNI, Gillam, all3, EG)
        '''
        new_X = []
        # Check if corpus is valid
        if corpus == 'all3':  # Include every corpus
            return self.X
        if corpus not in self.list_of_corpora:
            print("Not a valid corpus")
            return -1
        front_index, back_index = self.get_data_index(corpus)
        for feature in self.X:
            # Last corpus so splicing notation different
            if corpus == 'Gillam' or corpus == 'EG':
                new_X.append(feature[front_index:])
            else:
                new_X.append(feature[front_index:back_index])
        return new_X

    def get_data_index(self, corpus):
        # Get the index values of the corpus in X based on its implicit
        # structure
        corpus_index = self.list_of_corpora.index(corpus)*2
        if corpus == 'Conti4':
            front_index = 0
            back_index = sum(self.len_of_corpora[:corpus_index+2])
        elif corpus == 'ENNI':
            front_index = sum(self.len_of_corpora[:corpus_index])
            back_index = sum(self.len_of_corpora[:corpus_index+2])
        elif corpus == 'Gillam':  # Is Gillam
            front_index = sum(self.len_of_corpora[:corpus_index])
            back_index = None
        else:  # Gillam + ENNI together (age matched)
            corpus_index = 2  # ENNI
            front_index = sum(self.len_of_corpora[:corpus_index])
            back_index = None
        return front_index, back_index

    def get_Y_data(self, corpus='all3'):
        '''
        Returns the labels corresponding to the correct corpus
        Arguments:
            corpus - A string naming the corpus to extract
                     (Conti4, ENNI, Gillam, all3, EG)
        '''
        new_Y = []
        # Check if corpus is valid
        if corpus == 'all3':  # Include every corpus
            return self.Y
        if corpus not in self.list_of_corpora:
            print("Not a valid corpus")
            return -1
        front_index, back_index = self.get_data_index(corpus)
        if corpus == 'Gillam' or corpus == 'EG':
            new_Y.append(self.Y[front_index:])
        else:
            new_Y.append(self.Y[front_index:back_index])
        # Flatten the list
        new_Y = list(itertools.chain.from_iterable(new_Y))
        return new_Y

    def make_y(self):
        '''
        Populates the self.Y labels
        Is reliant on the form of self.corpora to be Corpus1 SLI -> Corpus 1 TD
        -> Corpus 2 SLI -> Corpus 3 TD -> ... -> Corpus n TD
        '''
        Y = []
        i = 1  # Index to determine if SLI or TD (SLI = odd) (TD = even)
        for corpus in self.corpora:
            if i % 2 == 0:  # is TD
                listofnums = [0] * len(corpus.fileids())
            else:  # is SLI
                listofnums = [1] * len(corpus.fileids())
            Y.append(listofnums)
            i += 1
        # Flatten the list
        Y = list(itertools.chain.from_iterable(Y))
        # Serialize the list
        try:
            y = open("y.txt", "w")
        except IOError:
            print('Could not open y data file')
        json.dump(Y, y, ensure_ascii=False)
        return Y

    def make_dataset(self):
        '''
        Makes the dataset to pass to SLIssifier.
        '''
        # Start creating features
        child_TNW = self.get_child_TNW()
        child_TNS = self.get_child_TNS()
        examiner_TNW = self.get_examiner_TNW()
        total_syl, average_syl = self.get_total_av_syl()
        f_k = self.get_flesch_kincaid(child_TNW, child_TNS, total_syl)
        r_2_i_verbs = self.raw_2_inflected_v()
        num_pos_tags = self.get_n_pos_tags()
        n_dos = self.get_n_dos()
        n_v, n_aux, n_3s_v, det_n_pl, det_pl_n, pro_aux, pro_3s_v, \
            total_error = self.get_error_rates()
        fillers = self.get_n_fillers()

        # Read in the CSV file as a Pandas Dataframe
        try:
            df_all_corpus = pd.read_csv('final_all_ages.csv')
        except IOError:
            print('Cannot open one of the CSV files')

        # Read the precomputed values from the CSV
        # [df_all_corpus[feature].tolist() \
        # for feature in list(df_all_corpus.columns.values)]
        mlu_words = df_all_corpus['MLU Words'].tolist()
        mlu_morphemes = df_all_corpus['MLU Morphemes'].tolist()
        mlu100_utts = df_all_corpus['MLU100 Utts'].tolist()
        freq_ttr = df_all_corpus['FREQ TTR'].tolist()  # Type/token ratio
        # vocd_d_opt_av = df_all_corpus['VOCD D_optimum_average'].tolist()
        verb_utt = df_all_corpus['Verbs/Utt'].tolist()  # Verbs per utterance
        td_utts = df_all_corpus['TD Utts'].tolist()  # Verbs per utterance
        word_errors = df_all_corpus['Word Errors'].tolist()
        retracing = df_all_corpus['retracing[//]'].tolist()
        repetition = df_all_corpus['repetition[/]'].tolist()
        dss = df_all_corpus['DSS'].tolist()  # Developmental Sentence Score
        ipsyn_total = df_all_corpus['IPSyn Total'].tolist()
        mor_words = df_all_corpus['mor Words'].tolist()
        # Browns grammatical morphemes
        present_progressive = df_all_corpus['*-PRESP'].tolist()
        propositions_in = df_all_corpus['in'].tolist()
        propositions_on = df_all_corpus['on'].tolist()
        plural_s = df_all_corpus['*-PL'].tolist()
        irregular_past_tense = df_all_corpus['*&PAST'].tolist()
        possessive_s = df_all_corpus['~poss|*'].tolist()
        uncontractible_copula = df_all_corpus['cop|*'].tolist()
        articles = df_all_corpus['det'].tolist()
        regular_past_ed = df_all_corpus['*-PAST'].tolist()
        regular_3rd_person_s = df_all_corpus['*-3S'].tolist()
        irregular_3rd_person = df_all_corpus['*&3S'].tolist()
        uncontractible_aux = df_all_corpus['aux|*'].tolist()
        contractible_copula = df_all_corpus['~cop|*'].tolist()
        contractible_aux = df_all_corpus['~aux|*'].tolist()
        # n-gram probabilities
        s_1g_log = df_all_corpus['s-1-log'].tolist()
        s_1g_ppl = df_all_corpus['s-1-ppl'].tolist()
        s_1g_ppl2 = df_all_corpus['s-1-ppl2'].tolist()
        s_2g_log = df_all_corpus['s-2-log'].tolist()
        s_2g_ppl = df_all_corpus['s-2-ppl'].tolist()
        s_2g_ppl2 = df_all_corpus['s-2-ppl2'].tolist()
        s_3g_log = df_all_corpus['s-3-log'].tolist()
        s_3g_ppl = df_all_corpus['s-3-ppl'].tolist()
        s_3g_ppl2 = df_all_corpus['s-3-ppl2'].tolist()
        d_1g_log = df_all_corpus['d-1-log'].tolist()
        d_1g_ppl = df_all_corpus['d-1-ppl'].tolist()
        d_1g_ppl2 = df_all_corpus['d-1-ppl2'].tolist()
        d_2g_log = df_all_corpus['d-2-log'].tolist()
        d_2g_ppl = df_all_corpus['d-2-ppl'].tolist()
        d_2g_ppl2 = df_all_corpus['d-2-ppl2'].tolist()
        d_3g_log = df_all_corpus['d-3-log'].tolist()
        d_3g_ppl = df_all_corpus['d-3-ppl'].tolist()
        d_3g_ppl2 = df_all_corpus['d-3-ppl2'].tolist()

        # Make the features that need this data
        z_mlu_sli, z_mlu_td, z_we_sli, z_we_td, z_r2v_sli, z_r2v_td, \
            z_utts_sli, z_utts_td = self.get_standard_scores(mlu_words,
                                                             word_errors,
                                                             r_2_i_verbs,
                                                             td_utts)
        # LM probabilities
        '''
        try:
        lms = open("lms_out.txt", "r")
        except IOError:
        print('Could not open lms file')
        lms_list = json.load(lms)
        '''

        f_labels = [child_TNW, child_TNS, examiner_TNW, freq_ttr, r_2_i_verbs,
                    mor_words, num_pos_tags, n_dos, repetition, retracing,
                    fillers, s_1g_log, s_2g_log, s_3g_log, d_1g_log, d_2g_log,
                    d_3g_log, z_mlu_sli,
                    z_mlu_td, z_we_sli, z_we_td, z_r2v_sli, z_r2v_td,
                    z_utts_sli, z_utts_td, total_syl, average_syl, mlu_words,
                    mlu_morphemes, mlu100_utts, verb_utt, dss, ipsyn_total,
                    present_progressive, propositions_in, propositions_on,
                    plural_s, irregular_past_tense, possessive_s,
                    uncontractible_copula, articles, regular_past_ed,
                    regular_3rd_person_s, irregular_3rd_person,
                    uncontractible_aux, contractible_copula, contractible_aux,
                    word_errors, f_k, n_v, n_aux, n_3s_v, det_n_pl, det_pl_n,
                    pro_aux, pro_3s_v, total_error]

        # Add the features to X
        # GROUP 1: Language Productivity
        self.X.append(child_TNW)
        self.X.append(child_TNS)
        self.X.append(examiner_TNW)
        self.X.append(freq_ttr)
        # GROUP 2: Morphosyntactic skills
        self.X.append(r_2_i_verbs)
        self.X.append(mor_words)
        self.X.append(num_pos_tags)
        self.X.append(n_dos)
        # GROUP 4: Speech fluency
        self.X.append(repetition)
        self.X.append(retracing)
        self.X.append(fillers)
        # GROUP 5: Probabilities from LMs
        self.X.append(s_1g_log)
        self.X.append(s_2g_log)
        self.X.append(s_3g_log)
        self.X.append(d_1g_log)
        self.X.append(d_2g_log)
        self.X.append(d_3g_log)
        # GROUP 6: Standard scores
        self.X.append(z_mlu_sli)
        self.X.append(z_mlu_td)
        self.X.append(z_we_sli)
        self.X.append(z_we_td)
        self.X.append(z_r2v_sli)
        self.X.append(z_r2v_td)
        self.X.append(z_utts_sli)
        self.X.append(z_utts_td)
        # GROUP 7: Sentence complexity
        self.X.append(total_syl)
        self.X.append(average_syl)
        self.X.append(mlu_words)
        self.X.append(mlu_morphemes)
        self.X.append(mlu100_utts)
        self.X.append(verb_utt)
        self.X.append(dss)
        self.X.append(ipsyn_total)
        self.X.append(present_progressive)
        self.X.append(propositions_in)
        self.X.append(propositions_on)
        self.X.append(plural_s)
        self.X.append(irregular_past_tense)
        self.X.append(possessive_s)
        self.X.append(uncontractible_copula)
        self.X.append(articles)
        self.X.append(regular_past_ed)
        self.X.append(regular_3rd_person_s)
        self.X.append(irregular_3rd_person)
        self.X.append(uncontractible_aux)
        self.X.append(contractible_copula)
        self.X.append(contractible_aux)
        # GROUP 8: Error rates
        self.X.append(word_errors)
        self.X.append(f_k)
        self.X.append(n_v)
        self.X.append(n_aux)
        self.X.append(n_3s_v)
        self.X.append(det_n_pl)
        self.X.append(det_pl_n)
        self.X.append(pro_aux)
        self.X.append(pro_3s_v)
        self.X.append(total_error)

        '''
        Garbage features

        self.X.append(vocd_d_opt_av)
        self.X.append(ndw_100)
        self.X.append(s_1g_ppl)
        self.X.append(s_1g_ppl2)
        self.X.append(s_3g_ppl)
        self.X.append(total_utts)
        self.X.append(mlu_utts)
        self.X.append(s_3g_ppl2)
        self.X.append(s_2g_ppl)
        self.X.append(s_2g_ppl2)
        self.X.append(d_2g_ppl)
        self.X.append(ipsyn_utts)
        self.X.append(d_2g_ppl2)
        self.X.append(d_1g_ppl)
        self.X.append(d_1g_ppl2)
        self.X.append(d_3g_ppl)
        self.X.append(td_utts)
        self.X.append(dss_utts)
        self.X.append(d_3g_ppl2)
        '''

         Serialize self.X
        try:
            x_outfile = open("x.txt", "w")
        except IOError:
            print('Could not open x file')
        json.dump(self.X, x_outfile, ensure_ascii=False)

    def only_pos_tags(self):
        pos_corpora = []
        for corpus in self.tnt_sents_tagged:
            tag_corpus = []  # Len = len of corpus
            for transcript in corpus:  # This is double nested
                pos_transcript = []
                for trans2 in transcript:
                    for sent in trans2:
                        if sent:  # Some sents are empty :(
                            words, tags = map(list, zip(*sent))
                        pos_transcript.append(tags)
                tag_corpus.append(pos_transcript)
            pos_corpora.append(tag_corpus)
        return pos_corpora

    def get_n_fillers(self):
        # Get a list of fillers from searching the word database
        list_of_fillers = ['um', 'umm', 'uh', 'uhh',
                           'uhhuh', 'ah', 'aah', 'ahh']
        fillers = []
        for corpora in self.words:
            for transcript in corpora:
                n_fillers = 0.0
                for word in transcript:
                    if word in list_of_fillers:
                        n_fillers += 1.0
                fillers.append(n_fillers)
        return fillers

    def get_n_dos(self):
        '''
        Returns the number of auxilliary "do" used in the transcript
        '''
        n_dos = []
        for corpora in self.tnt_sents_tagged:
            for corpus in corpora:
                for transcript in corpus:
                    n_do = 0
                    for sents in transcript:
                        for word in sents:
                            pos_tag = word[1]
                            if pos_tag == 'DO':
                                n_do += 1
                    n_dos.append(float(n_do))
        return n_dos

    def get_ndw_approx(self, words):
        '''
        Returns an approximation of NDW100
        '''
        ndw = 0
        words_seen = []
        for word in words:
            if word not in words_seen:
                ndw += 1
                words_seen.append(word)
        return float(ndw)

    def compute_z_scores(self, feature):
        '''
        Returns the z_scores both TD and SLI for a feature given to it
        Arguments:
        feature - A feature as a list of values
        '''
        # Since these are flat lists the numbers of transcripts in each corpus
        # have to be defined in order to distinguish the groups
        # Prepare the lists accordingly
        feature_TD = []
        feature_SLI = []
        # Split the lists accordingly
        sli_or_td = 1
        front_index = 0
        back_index = 0
        for i in range(0, len(self.len_of_corpora)):
            back_index += self.len_of_corpora[i]
            # Due to silly splicing notation have to check this
            if i == (len(self.len_of_corpora)-1):
                feature_TD.append(feature[front_index:])
            elif sli_or_td % 2 == 1:  # Is SLI
                feature_SLI.append(feature[front_index:back_index])
            else:  # Is TD
                feature_TD.append(feature[front_index:back_index])
            front_index = back_index
            sli_or_td += 1
        # Flatten the lists then convert to numpy arrays
        feature_SLI = list(itertools.chain.from_iterable(feature_SLI))
        feature_TD = list(itertools.chain.from_iterable(feature_TD))
        # Numpy arrays
        np_feature_sli = np.array(feature_SLI)
        np_feature_td = np.array(feature_TD)
        # Calculate mean and standard deviation for each
        mean_feature_sli = np.mean(np_feature_sli)
        mean_feature_td = np.mean(np_feature_td)
        std_feature_sli = np.std(np_feature_sli)
        std_feature_td = np.std(np_feature_td)
        # Prepare the lists
        z_feature_sli = []
        z_feature_td = []
        # Calculate the z-scores
        for i in range(0, len(feature)):
            z_feature_sli.append((feature[i]-mean_feature_sli)/std_feature_sli)
            z_feature_td.append((feature[i]-mean_feature_td)/std_feature_td)
        return z_feature_sli, z_feature_td

    def get_standard_scores(self, mlu_words, ndw_100, ipsyn_total, td_utts):
        '''
        Returns the z-scores for each child based on the mean score of both
        the SLI and the TD groups.
        The measures used are MLU, NDW, IPSYN_Total, and Total Utterances.
        Each child will therefore have 8 corresponding z-scores
        '''
        # Since these are flat lists the numbers of transcripts in each corpus
        # have to be defined in order to distinguish the groups
        # Prepare the lists accordingly
        mlu_SLI, mlu_TD = self.compute_z_scores(mlu_words)
        ndw_SLI, ndw_TD = self.compute_z_scores(ndw_100)
        ipsyn_total_SLI, ipsyn_total_TD = self.compute_z_scores(ipsyn_total)
        td_utts_SLI, td_utts_TD = self.compute_z_scores(td_utts)

        return mlu_SLI, mlu_TD, ndw_SLI, ndw_TD, ipsyn_total_SLI,\
            ipsyn_total_TD, td_utts_SLI, td_utts_TD

    def get_error_rates(self):
        '''
        Calculates error rates based on the following bigram pos tags
        - Noun-verb
        - Noun-auxillary verb
        - Noun-3rd person verb
        - Determiner-plural noun
        - Plural determiner-noun
        - Personal pronoun-3rd person verb
        - Personal pronoun-auxilliary verb
        '''
        per_pronouns = ['PRP', 'PPSS', 'PP$', 'PP$$', 'PPL', 'PPLS', 'PPO']
        auxillary_verbs = ['DO', 'HV', 'MD', 'DOD', 'BEZ', 'BED']
        # Initiate the error lists
        n_v = []  # noun-verb error
        n_aux = []  # noun-auxilliary verb error
        n_3s_v = []  # noun-3rd person verb error
        det_n_pl = []  # determiner-plural noun error
        det_pl_n = []  # plural determiner-noun error
        pro_aux = []  # pronoun-auxillary verb error
        pro_3s_v = []  # personal pronoun-3rd person singular verb error
        t_error = []  # total errors combined
        for corpora in self.tnt_sents_tagged:
            for corpus in corpora:
                for transcript in corpus:
                    n_v_error = 0.0
                    n_aux_error = 0.0
                    n_3s_verb_error = 0.0
                    det_n_pl_error = 0.0
                    det_pl_n_error = 0.0
                    pp_3s_verb_error = 0.0
                    pp_aux_error = 0.0
                    total_error = 0.0
                    for sents in transcript:
                        # Create the bigram
                        bigram = list(self.find_ngrams(2, sents))
                        for item in bigram:  # is a tuple of tuples
                            pos_word1 = item[0][1]
                            pos_word2 = item[1][1]
                            if pos_word1 == 'NN':
                                if pos_word2 == 'VB':
                                    n_v_error += 1
                                    total_error += 1
                                if pos_word2 == 'VBZ':
                                    n_3s_verb_error += 1
                                    total_error += 1
                                if pos_word2 in auxillary_verbs:
                                    n_aux_error += 1
                                    total_error += 1
                            if pos_word1 == 'DT' or 'DTI':
                                if pos_word2 == 'NNS':
                                    det_n_pl_error += 1
                                    total_error += 1
                            if pos_word1 == 'DTS':
                                if pos_word2 == 'NN':
                                    det_pl_n_error += 1
                                    total_error += 1
                            if pos_word1 in per_pronouns:
                                if pos_word2 == 'VBZ':
                                    pp_3s_verb_error += 1
                                    total_error += 1
                                if pos_word2 in auxillary_verbs:
                                    pp_aux_error += 1
                                    total_error += 1
                    n_v.append(n_v_error)
                    n_aux.append(n_aux_error)
                    n_3s_v.append(n_3s_verb_error)
                    det_n_pl.append(det_n_pl_error)
                    det_pl_n.append(det_pl_n_error)
                    pro_aux.append(pp_3s_verb_error)
                    pro_3s_v.append(pp_aux_error)
                    t_error.append(total_error)
        return n_v, n_aux, n_3s_v, det_n_pl, det_pl_n, pro_aux, pro_3s_v, \
            t_error

    def get_n_pos_tags(self):
        '''
        Calculates the number of different pos tags per transcript
        '''
        num_pos_tags = []
        for corpora in self.tnt_sents_tagged:
            for corpus in corpora:
                for transcript in corpus:
                    n_found = 0
                    tags_found = []
                    for sent in transcript:
                        for word in sent:
                            pos = word[1]
                            if pos not in tags_found:
                                tags_found.append(pos)
                                n_found += 1
                    num_pos_tags.append(float(n_found))
        return num_pos_tags

    def get_words_and_ids(self):
        '''
        Returns all the words from each corpora as a list of list
        '''
        words = []
        corpus_ids = []
        for corpus in self.corpora:
            corpus_words = []
            for file in corpus.fileids():
                corpus_ids.append(file)
                corpus_words.append(corpus.words(file, speaker=['CHI']))
            words.append(corpus_words)
        return words, corpus_ids

    def get_sents(self):
        '''
        Returns all the sents from each corpora as a list of list
        '''
        sents = []
        for corpus in self.corpora:
            corpus_sents = []
            for file in corpus.fileids():
                corpus_sents.append(corpus.sents(file, speaker=['CHI']))
            sents.append(corpus_sents)
        return sents

    def raw_2_inflected_v(self):
        '''
        Children with LI appear to have particular difficulty with
        morphemes such as -ed, -s, be, and do. This leads to the usage
        of raw (i.e. uninflected or root) verbs instead of their inflected
        equivalent forms.
        '''
        r2v = []
        inflected_verbs = ['VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
        raw_verb = 'VB'
        # self.tnt_sents_tagged has an extra list layer or two due to JSON
        for corpora in self.tnt_sents_tagged:
            for corpus in corpora:  # Corpus
                for transcript in corpus:  # Transcript
                    n_raw_verbs = 0
                    n_inflected_verbs = 0
                    for sents in transcript:  # Sentence
                        for word in sents:  # Word layer
                            word_tag = word[1]
                            if word_tag == raw_verb:
                                n_raw_verbs += 1
                            elif word_tag in inflected_verbs:
                                n_inflected_verbs += 1
                    if n_inflected_verbs == 0:
                        n_inflected_verbs += 1  # To avoid division by zero
                    r2v.append(n_raw_verbs/float(n_inflected_verbs))
        return r2v

    def get_nsyl(self, word):
        '''
        Returns the approximate number of syllables in a word based on the
        written method outlined in
        https://www.howmanysyllables.com/howtocountsyllables
        It's not perfect but it's a decent appoximation
        Arguments:
        word - The word to count the number of syllables in
        '''
        vowels = ['a', 'e', 'i', 'o', 'u', 'y']
        diphongs = ['ea', 'ee', 'ai', 'ei', 'ou', 'oo', 'oi', 'ay', 'ey',
                    'oy', 'oa', 'ow', 'ie']
        triphongs = ['aye', 'ayo', 'oya', 'our', 'owe']
        silent_vowels = ['ca', 'te', 'ke', 'he', 'in', 'es', 'ue', 'ui', 'ua']
        num_vowels = 0
        lastWasVowel = False
        word_chars = list(word)
        is_dipthong = False
        for i in range(0, len(word_chars)):
            foundVowel = False
            for v in vowels:
                # dont count diphthongs and triphongs
                if v == word_chars[i] and is_dipthong:
                    seq = (word_chars[i-2], word_chars[i-1], word_chars[i])
                    triphong = ''.join(seq)
                    if triphong in triphongs:
                        num_vowels -= 1
                if v == word_chars[i] and lastWasVowel:
                    seq = (word_chars[i-1], word_chars[i])
                    diphong = ''.join(seq)
                    if diphong in diphongs:
                        num_vowels -= 1
                    foundVowel = True
                    lastWasVowel = True
                    is_dipthong = True
                elif v == word_chars[i] and not lastWasVowel:
                    num_vowels += 1
                    if i != len(word_chars)-1:  # Not last letter
                        seq_vowel_1st = (word_chars[i], word_chars[i+1])
                        seq_vowel_2nd = (word_chars[i-1], word_chars[i])
                        vowel_1st = ''.join(seq_vowel_1st)
                        vowel_2nd = ''.join(seq_vowel_2nd)
                        if vowel_1st in silent_vowels \
                                or vowel_2nd in silent_vowels:
                            num_vowels -= 1
                    foundVowel = True
                    lastWasVowel = True
                #if full cycle and no vowel found, set lastWasVowel to false
                if not foundVowel:
                    lastWasVowel = False
        # remove les or le if vowel behind l its usually silent
        if len(word_chars) > 2:
            if word_chars[-1] == 's' \
                    and word_chars[-2] == 'e'and word_chars[-3] == 'l':
                if word_chars[-4] in vowels:
                    num_vowels -= 1
                elif word_chars[-1] == 'l' and word_chars[-2] == 'e':
                    if word_chars[-3] in vowels:
                        num_vowels -= 1
        # remove silent e
        elif len(word_chars) > 1 and word_chars[-1] == 'e':
            num_vowels -= 1
        return float(num_vowels)

    def get_total_av_syl(self):
        '''
        Returns the total and the average number of syllables used by the child
        '''
        syl_total = []
        syl_av = []
        # Create a dictionary using a comprehension - this maps every character
        # from string.punctuation to None. Initialize a translation object
        # from it.
        translator = str.maketrans({key: None for key in string.punctuation})
        for corpus in self.words:
            for transcript in corpus:
                child_syls = []
                for word in transcript:
                    word = word.translate(translator)
                    num_syl = self.get_nsyl(word)
                    child_syls.append(num_syl)
                # Append the average and the total to the respective lists
                syl_total.append(float(sum(child_syls)))
                syl_av.append(sum(child_syls)/float(len(child_syls)))
        return syl_total, syl_av

    def get_flesch_kincaid(self, total_words, total_sents, total_syl):
        '''
        Returns the Flesch-Kincaid grade level score, which corresponds to the
        approximate grade level the child would need to be to understand
        the text
        Arguments:
        total_words - List of total number of words used in each transcript
        total_sents - List of total number of sents used in each transcript
        total_syl   - List of total number of syllables used in each
        transcript
        '''
        flesch_kincaid = []
        for i in range(0, len(total_words)):
            child_fk_score = 0.39*(total_words[i]/float(total_sents[i])) +\
                11.8*(total_syl[i]/float(total_words[i])) - 15.59
            flesch_kincaid.append(child_fk_score)
        return flesch_kincaid

    def get_child_TNW(self):
        '''
        Returns the total number of words used by the child
        '''
        TNW = []
        for corpus in self.words:
            for transcript in corpus:
                TNW.append(float(len(transcript)))
        return TNW

    def get_child_TNS(self):
        '''
        Returns the total number of sentances used by the child
        '''
        TNS = []
        for corpus in self.sents:
            for transcript in corpus:
                TNS.append(float(len(transcript)))
        return TNS

    def get_examiner_TNW(self):
        '''
        Returns the total number of words used by the examiner
        '''
        TNW = []
        for corpus in self.corpora:
            for file in corpus.fileids():
                file_info = corpus.corpus(file)[0]  # Returns a dict in a list
                file_corpus = file_info['Corpus']
                if file_corpus == "ENNI":
                    TNW.append(float(len(corpus.words(file, speaker='EXA'))))
                else:
                    TNW.append(float(len(corpus.words(file, speaker='INV'))))
        return TNW

    def find_ngrams(self, n, bag_of_words):
        '''
        Returns a list of n tuples corresponding to n-grams using iterables
        zip library
        Arguments:
        n - Corresponds to the length of the n gram to be created
        bag_of_words - Is a list of the words used by a single child
        '''
        return zip(*[bag_of_words[i:] for i in range(n)])

    def tnt_tagger(self):
        '''
        CHILDES tagger inadequate for some purposes, so tag with same method
        as in Gambani but trained on the Brown corpus instead of switchboard
        Extremely slow (several hours) so dump the list to JSON
        Tags correspond to those used in the Brown corpus
        '''
        training_words = brown.tagged_sents()
        tagger = tnt.TnT()
        tagger.train(training_words)
        self.tnt_sents_tagged = []
        try:
            pos_sents_outfile = open("tagged_data.txt", "w")
        except IOError:
            print('Could not open tagged data file')
        for corpus in self.sents:
            corpus_data = []
            for transcript in corpus:
                transcript_data = []
                transcript_data.append(tagger.tagdata(transcript))
                corpus_data.append(transcript_data)
            self.tnt_sents_tagged.append(corpus_data)
        # Dump the list to JSON format for later
        json.dump(self.tnt_sents_tagged, pos_sents_outfile, ensure_ascii=False)

# =============================================================================
# For unit testing
# =============================================================================

if __name__ == '__main__':
    childes = CHILDESdata()
    childes.make_dataset()
    childes.make_y()
    '''def train_LMS(self, corpus, n_gram):

    Returns the perplexity of a LM model based off the corpus given to it.
    and length of the n-gram to compute

    Arguments:
    corpus - The entire set of transcripts from a corpus (SLI and TD)
    n_gram - The n-gram to compute

    lms = []
    test_set = []
    # Split the corpus based on SLI or TD
    for group in corpus:
    # Organized by age so shuffle the list for a even spread on the
    # cut
    shuffle(group)

    # Train on 80% f the corpus and test on the rest
    spl = 80*len(group)/100
    train = group[:spl]
    test = group[spl:]
    test_set.append(test)

    # Create the LM
    fdist = nltk.FreqDist(w for w in train)
    estimator = WittenBellProbDist(fdist, 0.2)
    lm = NgramModel(n_gram, train, estimator=estimator)
    lms.append(lm)
    return lms, test_set
'''
