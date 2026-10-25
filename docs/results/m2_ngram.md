# M2 — n-gram baselines

Counts from 18,171,957 training tokens; scored on all 99,853 predicted validation tokens (728,876 bytes of text). Lower is better everywhere. Smoothing settings were picked on validation (which only flatters these baselines).

| model | smoothing | n-grams stored | loss/token | perplexity | bits/byte | Hindi | Hinglish |
|---|---|---:|---:|---:|---:|---:|---:|
| uniform | 1/V for every token | 0 | 9.011 | 8,192 | 1.781 | 1.572 | 3.738 |
| unigram | add-k, k=0.001 | 8,125 | 7.499 | 1,807 | 1.482 | 1.303 | 3.166 |
| bigram | add-k, k=0.03 | 1,721,304 | 4.980 | 146 | 0.984 | 0.917 | 1.612 |
| trigram | add-k, k=0.003 | 8,937,426 | 5.289 | 198 | 1.045 | 1.013 | 1.352 |
| unigram | JM, λ=(0.999) | 8,125 | 7.499 | 1,807 | 1.482 | 1.303 | 3.166 |
| bigram | JM, λ=(0.999, 0.9) | 1,721,304 | 4.973 | 144 | 0.983 | 0.915 | 1.617 |
| trigram | JM, λ=(0.999, 0.9, 0.5) | 8,937,426 | 4.437 | 85 | 0.877 | 0.838 | 1.239 |
| 4-gram | JM, λ=(0.999, 0.9, 0.5, 0.35) | 20,590,629 | 4.291 | 73 | 0.848 | 0.829 | 1.029 |

## For reference: general-purpose compressors

Prediction and compression are the same problem: with arithmetic coding, a model that assigns X bits/byte compresses the text to X/8 of its size. The compressors below only see the validation text itself; the models above learned from the training set.

| compressor | bits/byte | Hindi | Hinglish |
|---|---:|---:|---:|
| gzip -9 | 1.773 | 1.645 | 2.908 |
| bzip2 -9 | 1.293 | 1.154 | 2.522 |
| xz -9e | 1.450 | 1.320 | 2.685 |

## Samples (60 tokens, sampled from the model's distribution)

**bigram, add-k, k=0.03** — prompt `भारत की राजधानी`

> भारत की राजधानी ‘जिँر कमल देने वाले विषय धाराव48 धर्म 1961 / Ger par 4th conपूर्व में तनाववारी हिस्सेदारी पुलिस ने आग्रह किया और अनेक नाइट्रिक अनुवाद हुआ है और कई मतदाता और घटक दिए जाते हैं वे मिंग के अनुसार इन्हें 'न

**bigram, add-k, k=0.03** — prompt `yaar kal movie dekhne`

> yaar kal movie dekhne ke liye alarm set kare watching acheासित रहा होगा और लोगों को चुनौती मोर्चे रंगबोर कुमार शहीदा इमारतें लेकर सोलंकी Tuesday recurringभर के नमू कुर बताया गया। अरब अमीराताओं की सजा के अधिकांश नाटक, किन्तु 1925 रेलवे के विस्तृत है

**4-gram, JM, λ=(0.999, 0.9, 0.5, 0.35)** — prompt `भारत की राजधानी`

> भारत की राजधानी प्राग अगर अपने फिदीम।  देखनेग्रेशन टका बयान कि समस्त ब्रह्मास्त्र एक मैदान है। रोमन कब्रेक गेंदब kar paar kiye S ò, कोलार केसरी सिंह बारह वर्ष की आयु तक मात किहाबी पर एक कीट निकला फ्लोरी तथा

**4-gram, JM, λ=(0.999, 0.9, 0.5, 0.35)** — prompt `yaar kal movie dekhne`

> yaar kal movie dekhne he tuesday ko 645 pm me weather mil sakta hai
