# -*- coding: utf-8 -*-
"""

ngram_lms
~~~~~~~~~~~~~~~

Creates a text corpora of the PoS sentences for each child

Also checks if the file ids match

"""

import nltk
from nltk.corpus.reader import CHILDESCorpusReader
import pandas as pd
import os
import csv
import json
import subprocess


class CreateLMs():
    def __init__(self):
        self.X = []  # Where the features will go as list of lists
        self.Y = []  # Index corresponds to the label (SLI or TD)

        # Create the corpus readers for each corpus
        nltk.data.path.append(os.getcwd() + '/')

        # Add the CWD to the datapath
        corpus_root = nltk.data.find('Data/')
        self.sli_conti4 = CHILDESCorpusReader(corpus_root,
                                              'SLI/Conti4/.*.xml')
        self.td_conti4 = CHILDESCorpusReader(corpus_root,
                                             'TD/Conti4/.*.xml')
        self.sli_enni = CHILDESCorpusReader(corpus_root,
                                            'SLI/ENNI/.*.xml')
        self.td_enni = CHILDESCorpusReader(corpus_root,
                                           'TD/ENNI/.*.xml')
        self.sli_gil = CHILDESCorpusReader(corpus_root,
                                           'SLI/Gillam/.*.xml')
        self.td_gil = CHILDESCorpusReader(corpus_root,
                                          'TD/Gillam/.*.xml')
        self.corpora = [self.sli_conti4, self.td_conti4, self.sli_enni,
                        self.td_enni, self.sli_gil, self.td_gil]
        # Check if file_ids match to CSV file
        nltk_ids = [corpus.fileids() for corpus in self.corpora]
        nltk_ids = [os.path.basename(filename)
                    for corpus in nltk_ids for filename in corpus]
        self.nltk_ids = [os.path.splitext(base)[0] for base in nltk_ids]
        df_all = pd.read_csv('final_all_ages.csv')
        csv_ids = df_all['File'].tolist()
        csv_ids = [os.path.splitext(base)[0] for base in csv_ids]
        print(csv_ids == self.nltk_ids)

        # Get the sentences
        try:
            pos_sents_tnt = open("tagged_data.txt", "r")
        except IOError:
            print('Could not open tagged data file')
        tnt_sents_tagged = json.load(pos_sents_tnt)

        # Write a corpus for each file
        self.pos_corpora = self.only_pos_tags(tnt_sents_tagged)
        self.corpora_names = ['sli_conti4', 'td_conti4', 'sli_enni',
                              'td_enni', 'sli_gil', 'td_gil']

    def training_files(self):
        '''
        Writes a text copy of each corpus by group. Then writes a copy of
        each corpus without the transcript.
        '''
        # Create the training files for each corpus
        index = 0
        for corpus in self.pos_corpora:
            for transcript in corpus:
                my_path = os.getcwd() + '/%s' % self.corpora_names[index]
                target = open(my_path + '.Train', 'w')
                for sent in transcript:
                    s = ' '
                    target.write(s.join(sent) + '\n')
            index += 1

        # Create the training corpora for individual transcripts
        index = 0
        tran_index = 0
        for corpus in self.pos_corpora:
            for transcript in corpus:
                exclude_trans = [x for i, x in enumerate(corpus)
                                 if i != transcript]
                my_path = os.getcwd() + '/%s-Train' % self.corpora_names[index]
                if not os.path.isdir(my_path):
                    os.makedirs(my_path)
                target = open(my_path +
                              '/%s.Train' % self.nltk_ids[tran_index], 'w')
                for transcript in exclude_trans:
                    for sent in transcript:
                        s = ' '
                        target.write(s.join(sent) + '\n')
                tran_index += 1
            index += 1
        # Write individual transcripts to file
        corp_index = 0
        tran_index = 0
        for corpus in self.pos_corpora:
            my_path = os.getcwd() + '/%s-ind_trans/' \
                % self.corpora_names[corp_index]
            if not os.path.isdir(my_path):
                os.makedirs(my_path)
            for transcript in corpus:
                target = open(os.getcwd() + '/%s-ind_trans/%s.txt' %
                              (self.corpora_names[corp_index],
                               self.nltk_ids[tran_index]), 'w')
                for sent in transcript:
                    s = ' '
                    target.write(s.join(sent) + '\n')
                tran_index += 1
            corp_index += 1

    def create_lms(self):
        '''
        Creates the LMs for each corpus
        '''
        # Write the LMs for each transcript's corpus
        for corpus in self.corpora_names:
            my_path = os.getcwd() + '/%s-Train' % corpus
            for filename in os.listdir(my_path):
                filename = my_path + '/' + os.path.splitext(filename)[0]
                # Create the lms for each corpus
                self.ngram_count_counts(filename)
                self.ngram_count_lms(filename, 3)
        # Write LMs for each entire corpus
        for corpus in self.corpora_names:
            my_path = os.getcwd() + '/%s' % corpus
            self.ngram_count_counts(my_path)
            self.ngram_count_lms(my_path, 3)

    def ngram_count_lms(self, filename, order):
        '''
        Creates the LMs with witten-bell discounting using SRILM
        '''
        subprocess.Popen("ngram-count -read %s -order %d -lm %s "
                         "-wbdiscount1 -wbdiscount2 -wbdiscount3"
                         % (filename + '.count',
                            order,
                            filename + '.lm-' + str(order)),
                         shell=True, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT)

    def ngram_count_counts(self, filename):
        '''
        Creates the 1,2,3-gram counts using SRILM
        '''
        subprocess.Popen('ngram-count -text %s -order 3 -write %s -unk'
                         % (filename + '.Train', filename + '.count'),
                         shell=True, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT)

    def ngram_perplexity(self, train_file, lm_file, order):
        '''
        Returns the probability that the transcript belongs to the
        training set
        @params:
            train_file - full file name directory to the text to compute the
                         probability from
            lm_file - full file name directory of the LM (sans the file
                      extension)
            order - the n-gram LM to compute
        '''
        p = subprocess.Popen('ngram -ppl %s -order %d -lm %s'
                             % (train_file,
                                order,
                                lm_file + '.lm-3'),
                             shell=True, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)
        l = []
        try:
            for t in p.stdout.readlines()[1].split():
                try:
                    l.append(float(t))
                except ValueError:
                    pass
        except IndexError:
            print(p.stdout.readlines())

        return l

    def compute_perplexity(self):
        '''
        Computes the probablity that the transcript belongs to that corpus
        '''
        # Compute perplexity
        s = {}
        d = {}
        for corpus in self.corpora_names:
            my_path = os.getcwd() + '/%s-ind_trans' % corpus
            for filename in os.listdir(my_path):
                train_file = my_path + '/' + filename
                # Compute the perplexity for the same group
                lm_file = os.getcwd() + '/%s-Train/%s' \
                    % (corpus, os.path.splitext(filename)[0])
                for order in range(1, 4):
                    pplxity = self.ngram_perplexity(train_file, lm_file, order)
                    pplxity.append(corpus)
                    if pplxity[1] == 0.0:
                        print(train_file)
                        print(lm_file + '-' + str(order))
                    if filename in s.keys():
                        s[filename].append(pplxity)
                    else:
                        s[filename] = [pplxity]
                # Compute the perplexity against the opposing group
                if 'sli' in corpus:
                    op = self.corpora_names[self.corpora_names.index(corpus)+1]
                else:
                    op = self.corpora_names[self.corpora_names.index(corpus)-1]
                lm_file = os.getcwd() + '/%s' % op
                for order in range(1, 4):
                    pplxity = self.ngram_perplexity(train_file, lm_file, order)
                    pplxity.append(op)
                    if pplxity[1] == 0.0:
                        print(train_file)
                        print(lm_file + str(order))
                    if filename in d.keys():
                        d[filename].append(pplxity)
                    else:
                        d[filename] = [pplxity]
        # Write to file
        with open('s.csv', 'w') as csv_file:
            writer = csv.writer(csv_file)
            for key, val in s.items():
                writer.writerow([key,
                                 val[0][4], val[0][1], val[0][2], val[0][3],
                                 val[1][1], val[1][2], val[1][3],
                                 val[2][1], val[2][2], val[2][3]])
        with open('d.csv', 'w') as _file:
            writer = csv.writer(_file)
            for key, val in d.items():
                writer.writerow([key,
                                 val[0][4], val[0][1], val[0][2], val[0][3],
                                 val[1][1], val[1][2], val[1][3],
                                 val[2][1], val[2][2], val[2][3]])

    def only_pos_tags(self, tags):
        pos_corpora = []
        for corpus in tags:
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

if __name__ == '__main__':
    lms = CreateLMs()
    # lms.training_files()
    # lms.create_lms()
    lms.compute_perplexity()
