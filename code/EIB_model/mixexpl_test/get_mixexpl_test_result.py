import json

results = {
    'all_test_pred': [],
    'all_test_target': [],
    'all_task_sample': [],
    'e_delta_nli': [],
    'e_delta_nli_target': [],
    'e_delta_nli_task_sample': [],
    'liar_plus': [],
    'liar_plus_target': [],
    'liar_plus_task_sample': [],
    'pubhealth': [],
    'pubhealth_target': [],
    'pubhealth_task_sample': [],
    'science_qa': [],
    'science_qa_target': [],
    'science_qa_task_sample': [],
    'senmaking': [],
    'senmaking_target': [],
    'senmaking_task_sample': [],
}

for name in ['pubhealth', 'e_delta_nli', 'liar_plus', 'science_qa', 'senmaking']:
    file = open('{}/prediction_test.txt'.format(name))
    task_sample = ''
    for line in file.readlines():
        if 'prediction_x' in line:
            line = line.replace('prediction_x: ', '')
            results[name].append(line)
            results['all_test_pred'].append(line)
        elif 'expl_opt:' in line:
            line = line.replace('expl_opt: ', '')
            results[name+'_target'].append(line)
            results['all_test_target'].append(line)
        elif 'sample_in:' in line:
            line = line.replace('sample_in: ', '').replace('<|exp|>', ' ').strip('\n').strip('') + ' '
            task_sample += line
        elif 'sample_out:' in line:
            line = line.replace('sample_out: ', '').replace('<|exp|>', ' ').strip('\n').strip('')
            task_sample += line
            results[name+'_task_sample'].append(task_sample)
            results['all_task_sample'].append(task_sample)
            task_sample = ''

json.dump(results, open('new_mixexpl_test_results.json', 'w'), indent=4)





'''
all_test:  828
avg lens:  43.02777777777778
bleu1:  0.6547023580497994
bleu2:  0.6258021267212617
bleu4:  0.5845833157524172
sacrebleu:  BLEU = 0.11 4.2/0.1/0.0/0.0 (BP = 1.000 ratio = 51.062 hyp_len = 817
 ref_len = 16)
dist1:  16.171054801297203
dist2:  40.22723437763654
Some weights of the model checkpoint at roberta-large were not used when initial
izing RobertaModel: ['lm_head.bias', 'lm_head.layer_norm.weight', 'lm_head.decod
er.weight', 'lm_head.layer_norm.bias', 'lm_head.dense.weight', 'lm_head.dense.bi
as']
- This IS expected if you are initializing RobertaModel from the checkpoint of a
 model trained on another task or with another architecture (e.g. initializing a
 BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing RobertaModel from the checkpoint
of a model that you expect to be exactly identical (initializing a BertForSequen
ceClassification model from a BertForSequenceClassification model).
BERTScore: 0.93906
CIDEr
{'Bleu_1': 0.661408482330798, 'Bleu_2': 0.6324196066359957, 'Bleu_3': 0.61068352
13927245, 'Bleu_4': 0.5915080417303126, 'METEOR': 0.41142614737956135, 'ROUGE_L'
: 0.6831407979568632, 'CIDEr': 3.594073170351473, 'SkipThoughtCS': 0.84511024}
novelty 1 (sample avg):  0.6858499221990779
novelty 2 (sample avg):  0.8746298416666552
novelty 1 (corpus):  0.40567257679818386
novelty 2 (corpus):  0.3610735940687951
novelty 1 avg:  0.5457612494986309
novelty 2 avg:  0.6178517178677252


pubhealth: 177
avg lens:  49.621468926553675
bleu1:  0.6639763555757645
bleu2:  0.6380963400315488
bleu4:  0.6011616130946955
sacrebleu:  BLEU = 0.04 2.6/0.0/0.0/0.0 (BP = 1.000 ratio = 57.385 hyp_len = 298
4 ref_len = 52)
dist1:  26.054336705695125
dist2:  50.82366589327146
Some weights of the model checkpoint at roberta-large were not used when initial
izing RobertaModel: ['lm_head.bias', 'lm_head.layer_norm.weight', 'lm_head.decod
er.weight', 'lm_head.layer_norm.bias', 'lm_head.dense.weight', 'lm_head.dense.bi
as']
- This IS expected if you are initializing RobertaModel from the checkpoint of a
 model trained on another task or with another architecture (e.g. initializing a
 BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing RobertaModel from the checkpoint
of a model that you expect to be exactly identical (initializing a BertForSequen
ceClassification model from a BertForSequenceClassification model).
BERTScore: 0.94252
CIDEr
{'Bleu_1': 0.6698166913354583, 'Bleu_2': 0.6443232718808951, 'Bleu_3': 0.6249629
808420365, 'Bleu_4': 0.6079122341512627, 'METEOR': 0.4434663943831339, 'ROUGE_L'
: 0.6957916931143074, 'CIDEr': 3.8788698663997474, 'SkipThoughtCS': 0.8432198}
novelty 1 (sample avg):  0.7747324593742101
novelty 2 (sample avg):  0.9323538844872784
novelty 1 (corpus):  0.5049965140599582
novelty 2 (corpus):  0.4800139437601673
novelty 1 avg:  0.6398644867170842
novelty 2 avg:  0.7061839141237228

e_delta_nli: 134
avg lens:  37.85820895522388
bleu1:  0.7562081198265668
bleu2:  0.7230237240282208
bleu4:  0.681564650495963
sacrebleu:  BLEU = 0.02 2.4/0.0/0.0/0.0 (BP = 1.000 ratio = 41.065 hyp_len = 505
1 ref_len = 123)
dist1:  14.071738273551437
dist2:  32.79352226720648
Some weights of the model checkpoint at roberta-large were not used when initial
izing RobertaModel: ['lm_head.bias', 'lm_head.layer_norm.weight', 'lm_head.decod
er.weight', 'lm_head.layer_norm.bias', 'lm_head.dense.weight', 'lm_head.dense.bi
as']
- This IS expected if you are initializing RobertaModel from the checkpoint of a
 model trained on another task or with another architecture (e.g. initializing a
 BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing RobertaModel from the checkpoint
of a model that you expect to be exactly identical (initializing a BertForSequen
ceClassification model from a BertForSequenceClassification model).
BERTScore: 0.94453
CIDEr
{'Bleu_1': 0.7565543071159556, 'Bleu_2': 0.723368298520351, 'Bleu_3': 0.70076254
01167176, 'Bleu_4': 0.6819110346166873, 'METEOR': 0.4857481154825198, 'ROUGE_L':
 0.7805060434312251, 'CIDEr': 5.0506734886371065, 'SkipThoughtCS': 0.8986469}
novelty 1 (sample avg):  0.3932638573081776
novelty 2 (sample avg):  0.6100037764852606
novelty 1 (corpus):  0.3265843288115003
novelty 2 (corpus):  0.22393197003441992
novelty 1 avg:  0.35992409305983897
novelty 2 avg:  0.41696787325984025


senmaking: 177
avg lens:  13.847457627118644
bleu1:  0.4549800796812749
bleu2:  0.42861194793379154
bleu4:  0.37363181793151207
sacrebleu:  BLEU = 0.26 1.9/0.2/0.1/0.1 (BP = 1.000 ratio = 13.500 hyp_len = 216 ref_len = 16)
dist1:  28.844621513944226
dist2:  51.778825546506646
Some weights of the model checkpoint at roberta-large were not used when initializing RobertaModel: ['lm_head.bias', 'lm_head.layer_norm.weight', 'lm_head.decoder.weight', 'lm_head.layer_norm.bias', 'lm_head.dense.weight', 'lm_head.dense.bias']
- This IS expected if you are initializing RobertaModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing RobertaModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
BERTScore: 0.94397
CIDEr
{'Bleu_1': 0.46878824969381117, 'Bleu_2': 0.44137632345835176, 'Bleu_3': 0.4142161458982189, 'Bleu_4': 0.38539539878289136, 'METEOR': 0.47274395413662035, 'ROUGE_L': 0.6832412919820573, 'CIDEr': 4.439488203056386, 'SkipThoughtCS': 0.7641686}
novelty 1 (sample avg):  0.739569446574929
novelty 2 (sample avg):  0.9484178010396833
novelty 1 (corpus):  0.5030782761653474
novelty 2 (corpus):  0.4678979771328056
novelty 1 avg:  0.6213238613701382
novelty 2 avg:  0.7081578890862444


liar_plus: 239
avg lens:  53.89121338912134
bleu1:  0.6009291335534895
bleu2:  0.5740388453343427
bleu4:  0.5361657259213786
sacrebleu:  BLEU = 0.04 2.8/0.0/0.0/0.0 (BP = 1.000 ratio = 65.162 hyp_len = 241
1 ref_len = 37)
dist1:  22.126148382614065
dist2:  50.00786534528866
Some weights of the model checkpoint at roberta-large were not used when initial
izing RobertaModel: ['lm_head.bias', 'lm_head.layer_norm.weight', 'lm_head.decod
er.weight', 'lm_head.layer_norm.bias', 'lm_head.dense.weight', 'lm_head.dense.bi
as']
- This IS expected if you are initializing RobertaModel from the checkpoint of a
 model trained on another task or with another architecture (e.g. initializing a
 BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing RobertaModel from the checkpoint
of a model that you expect to be exactly identical (initializing a BertForSequen
ceClassification model from a BertForSequenceClassification model).
BERTScore: 0.92874
CIDEr
{'Bleu_1': 0.6080211563262133, 'Bleu_2': 0.5811813942920893, 'Bleu_3': 0.5612440
001332534, 'Bleu_4': 0.5437811629014726, 'METEOR': 0.3683058155121009, 'ROUGE_L'
: 0.6229461708028515, 'CIDEr': 2.0804091879339865, 'SkipThoughtCS': 0.86712235}
novelty 1 (sample avg):  0.7659053868498518
novelty 2 (sample avg):  0.9152526281704916
novelty 1 (corpus):  0.4960050628905941
novelty 2 (corpus):  0.44854046357092003
novelty 1 avg:  0.6309552248702229
novelty 2 avg:  0.6818965458707058


science_qa: 101
avg lens:  63.76237623762376
bleu1:  0.5076574021554169
bleu2:  0.4825261712908905
bleu4:  0.4455611829048626
sacrebleu:  BLEU = 0.03 1.7/0.0/0.0/0.0 (BP = 1.000 ratio = 74.579 hyp_len = 283
4 ref_len = 38)
dist1:  10.280771412365286
dist2:  22.08315350309308
Some weights of the model checkpoint at roberta-large were not used when initial
izing RobertaModel: ['lm_head.bias', 'lm_head.layer_norm.weight', 'lm_head.decod
er.weight', 'lm_head.layer_norm.bias', 'lm_head.dense.weight', 'lm_head.dense.bi
as']
- This IS expected if you are initializing RobertaModel from the checkpoint of a
 model trained on another task or with another architecture (e.g. initializing a
 BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing RobertaModel from the checkpoint
of a model that you expect to be exactly identical (initializing a BertForSequen
ceClassification model from a BertForSequenceClassification model).
BERTScore: 0.92998
CIDEr
{'Bleu_1': 0.5083850931676229, 'Bleu_2': 0.4823473450083839, 'Bleu_3': 0.4626433
38510844, 'Bleu_4': 0.44503272120390985, 'METEOR': 0.4181160281405538, 'ROUGE_L'
: 0.6740574030154507, 'CIDEr': 2.8116586679789446, 'SkipThoughtCS': 0.86715424}
novelty 1 (sample avg):  0.63468865239038
novelty 2 (sample avg):  0.8991191720786667
novelty 1 (corpus):  0.24167849818583373
novelty 2 (corpus):  0.2285849503076195
novelty 1 avg:  0.43818357528810686
novelty 2 avg:  0.563852061193143
'''