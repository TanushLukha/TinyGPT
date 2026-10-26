# M6 — what the language model knows: 20 capitals

`scripts/probe_facts.py`. Prompt: `<place> की राजधानी` ("the capital of <place>"). *Rank*: where the model's likelihood puts the right city among all 20; *pick*: its first choice; *reverse*: the right place's rank the other way round, after `<city>` in `<city> <place> की राजधानी है`; *greedy*: the text it writes at temperature 0; *demo*, *min-p*: how many of 10 samples name the right city in their first three words, with the demo's default settings (temperature 0.8, top-k 50) or temperature 0.7 and min-p 0.1; *fact*, *city*: how often the training text contains `<place> की राजधानी <city>`, and the city at all.

The right capital ranks first for **2 of 20** places (chance: 1 in 20); its average rank is 9.7 of 20 (chance: 10.5). The greedy text names it for 1 of 20 places; samples name it 4 times in 200 with the demo's settings (temperature 0.8, top-k 50) and 6 in 200 with temperature 0.7, min-p 0.1. The other way round (`<city> ___ की राजधानी है`: which place?), the right place ranks first for 2 of 20, average rank 7.1.

| prompt | right city | rank | pick | reverse | greedy | demo | min-p | fact | city |
|---|---|---:|---|---:|---|---:|---:|---:|---:|
| भारत की राजधानी | दिल्ली | 1 | दिल्ली | 1 | दिल्ली में स्थित एक … | 0/10 | 2/10 | 23 | 4,191 |
| महाराष्ट्र की राजधानी | मुंबई | 4 | दिल्ली | 9 | है। यह शहर 2001 … | 0/10 | 0/10 | 3 | 1,305 |
| बिहार की राजधानी | पटना | 1 | पटना | 10 | है। यह शहर 2001 … | 1/10 | 1/10 | 15 | 867 |
| उत्तर प्रदेश की राजधानी | लखनऊ | 3 | पटना | 5 | पटना में स्थित एक … | 1/10 | 1/10 | 8 | 717 |
| राजस्थान की राजधानी | जयपुर | 3 | दिल्ली | 3 | दिल्ली में स्थित एक … | 1/10 | 1/10 | 7 | 902 |
| मध्य प्रदेश की राजधानी | भोपाल | 10 | पटना | 5 | पटना में स्थित एक … | 0/10 | 0/10 | 0 | 426 |
| पश्चिम बंगाल की राजधानी | कोलकाता | 7 | पटना | 5 | है। यह उत्तर प्रदेश … | 0/10 | 0/10 | 2 | 818 |
| तमिलनाडु की राजधानी | चेन्नई | 8 | पटना | 18 | है। यह उत्तर प्रदेश … | 0/10 | 0/10 | 0 | 549 |
| गुजरात की राजधानी | गांधीनगर | 17 | दिल्ली | 11 | दिल्ली में स्थित एक … | 0/10 | 0/10 | 1 | 30 |
| पंजाब की राजधानी | चंडीगढ़ | 7 | पटना | 10 | पटना में स्थित एक … | 0/10 | 0/10 | 2 | 313 |
| ओडिशा की राजधानी | भुवनेश्वर | 11 | दिल्ली | 14 | है। यह शहर 2001 … | 0/10 | 0/10 | 0 | 156 |
| असम की राजधानी | दिसपुर | 20 | दिल्ली | 14 | , जिसे 1972 में … | 0/10 | 0/10 | 1 | 11 |
| केरल की राजधानी | तिरुवनंतपुरम | 19 | दिल्ली | 3 | है। यह 2001 में … | 0/10 | 0/10 | 1 | 51 |
| झारखंड की राजधानी | रांची | 12 | पटना | 11 | है। यह शहर 2001 … | 0/10 | 0/10 | 5 | 218 |
| नेपाल की राजधानी | काठमांडू | 7 | दिल्ली | 3 | दिल्ली में स्थित एक … | 0/10 | 0/10 | 3 | 97 |
| पाकिस्तान की राजधानी | इस्लामाबाद | 13 | दिल्ली | 5 | है। यह 2001 में … | 0/10 | 0/10 | 5 | 67 |
| जापान की राजधानी | टोक्यो | 15 | दिल्ली | 5 | है। यह शहर 2001 … | 0/10 | 0/10 | 1 | 160 |
| फ्रांस की राजधानी | पेरिस | 10 | दिल्ली | 1 | है। यह 2001 में … | 0/10 | 0/10 | 5 | 429 |
| चीन की राजधानी | बीजिंग | 17 | दिल्ली | 4 | है। यह 2001 में … | 0/10 | 0/10 | 4 | 88 |
| रूस की राजधानी | मास्को | 9 | दिल्ली | 5 | है। यह 2001 में … | 1/10 | 1/10 | 2 | 136 |
