# -*- coding: utf-8 -*-
#!/usr/bin/env python
from layers.file_output_layer import FileOutputLayer
from layers.knowledge_layer import KnowledgeGroupLayer
from layers.mask_layer import MaskLayer
from layers.tokenize_group_layer import TokenizeGroupLayer
from layers.dict_group_layer import DictGroupLayer

import sys
from evaluator import evaluator
import os
import re
import string
import hashlib
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import argparse

input_dir = '../logs/' # The input directory of log file
output_dir = 'LogParserResult/' # The output directory of parsing results

def load_logs(log_file, regex, headers):
    """ Function to transform log file to dataframe
    """
    log_messages = dict()
    linecount = 0
    with open(log_file, 'r') as fin:
        for line in tqdm(fin.readlines(), desc='load data'):
            try:
                linecount += 1
                match = regex.search(line.strip())
                message = dict()
                for header in headers:
                    message[header] = match.group(header)
                message['LineId'] = linecount
                log_messages[linecount] = message
            except Exception as e:
                pass
    return log_messages

def generate_logformat_regex(logformat):
    """ Function to generate regular expression to split log messages
    """
    headers = []
    splitters = re.split(r'(<[^<>]+>)', logformat)
    regex = ''
    for k in range(len(splitters)):
        if k % 2 == 0:
            splitter = re.sub(' +', '\\\s+', splitters[k])
            regex += splitter
        else:
            header = splitters[k].strip('<').strip('>')
            regex += '(?P<%s>.*?)' % header
            headers.append(header)
    regex = re.compile('^' + regex + '$')
    return headers, regex

benchmark_settings = {
    'HDFS': {
        'log_file': 'HDFS/HDFS_2k.log',
        'log_format': '<Date> <Time> <Pid> <Level> <Component>: <Content>',
        'regex': [r'blk_-?\d+', r'(\d+\.){3}\d+(:\d+)?']
        },

    'BGL': {
        'log_file': 'BGL/BGL_2k.log',
        'log_format': '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
        'regex': [r'core\.\d+']
        }
}

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dictionary', default='../EngCorpus.pkl', type=str)
    args = parser.parse_args()
    corpus = args.dictionary

    benchmark_result = []
    for dataset, setting in benchmark_settings.items():
        print('\n=== Evaluation on %s ==='%dataset)
        indir = os.path.join(input_dir, os.path.dirname(setting['log_file']))
        outdir = os.path.join(output_dir, os.path.dirname(setting['log_file']))
        log_file = os.path.basename(setting['log_file'])

        filepath = os.path.join(indir, log_file)
        print('Parsing file: ' + filepath)
        starttime = datetime.now()
        headers, regex = generate_logformat_regex(setting['log_format'])
        log_messages = load_logs(filepath, regex, headers)
        # preprocess layer
        log_messages = KnowledgeGroupLayer(log_messages).run()
        # tokenize layer
        log_messages = TokenizeGroupLayer(log_messages, rex=setting['regex']).run()
        # dictionarize layer and cluster by wordset
        dict_group_result = DictGroupLayer(log_messages, corpus).run()
        # apply LCS and prefix tree
        results, templates = MaskLayer(dict_group_result).run()
        output_file = os.path.join(outdir, log_file)
        # output parsing results
        FileOutputLayer(log_messages, output_file, templates, ['LineId'] + headers).run()
        print('Parsing done. [Time taken: {!s}]'.format(datetime.now() - starttime))
        F1_measure, accuracy = evaluator.evaluate(
                            groundtruth=os.path.join(indir, log_file + '_structured.csv'),
                            parsedresult=os.path.join(outdir, log_file + '_structured.csv')
                            )
        benchmark_result.append([dataset, F1_measure, accuracy])

    print('\n=== Overall evaluation results ===')
    avg_accr = 0
    for i in range(len(benchmark_result)):
        avg_accr += benchmark_result[i][2]
    avg_accr /= len(benchmark_result)
    pd_result = pd.DataFrame(benchmark_result, columns=['dataset', 'F1_measure', 'Accuracy'])
    print(pd_result)
    print('avarage accuracy is {}'.format(avg_accr))
    pd_result.to_csv('benchmark_result.csv', index=False)
