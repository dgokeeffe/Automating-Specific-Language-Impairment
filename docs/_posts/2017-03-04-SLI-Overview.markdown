---
layout: default
title: Overview
date: 2017-02-04 00:21:00
categories: main
---

## So what is Specific Language Impairment??
Specific Language Impairment (SLI or specific language delay) is an interesting disorder that affects [roughly 7% of 5 year old children](https://mitpress.mit.edu/books/children-specific-language-impairment-0). It is characterized by a significant deficit in language ability in spite of any _obvious physical or mental disability_ such as hearing loss, neurological damage, and low nonverbal intelligence. 

I think this is best explained through an example so lets look at some real data from the [ENNI corpus](ENNI) hosted at the [CHILDES Talkbank](http://childes.psy.cmu.edu/). The children are asked to tell the same story from a wordless picture book about a young female elephant and young male giraffe at a swimming pool.

> #### 5yr-9mth old female Specific Language Impaired child 
> CHI: The elephant...s, see a garden  
> EXA: See what?  
> CHI: See a garden  
> CHI: See a see a water  
> CHI: And...s the elephant see a ball  

It's a little unclear what she's trying to convey in these sentences and the use of the singular determinant _a_ in front the word water is also clearly wrong.

> #### 5yr-8mth old female Typically Developing child 
> CHI: One, one day, um, a little giraffe and um, elephant they were playing with three balls  
> CHI: And they were, they were going to go in the swimming pool

While there is some stuttering and repeating the gist of the story is clear. In a nutshell all SLI means is that the child is delayed in their language development in comparison to their peers, but there's no obvious reason why.

## Diagnosing SLI through narratives alone is hard

It is better to start therapy with language delayed children sooner rather than later so a prompt diagnosis is the key to better outcomes. The problem is that the analysis of such narrative transcripts for a diagnosis is laborious for a speech pathologist or paediatrician to perform. This is where the wonders of NLP and Machine Learning come in. By extracting certain grammatical and morphosyntactic features from the text, it might be possible to build a supervised learning classifier to distinguish between SLI and TD children.

This idea was originally published back in 2011 by [Gabani et al.](https://www.ncbi.nlm.nih.gov/pubmed/?term=Gabani%20K%5BAuthor%5D&cauthor=true&cauthor_uid=21937203) from The University of Texas at Dallas. While my work borrowed heavily from their initial ideas, I thought that there was some room for improvement in some key areas.

Firstly, the [Conti-Ramsden 4](https://www.ncbi.nlm.nih.gov/pubmed/17729147) corpus their paper was build upon was short on sample size, with a total of 19 SLI and 99 TD children. The children also... weren't really children; they were aged 13, 14, and 15. While SLI suffers are certainly delayed, their language skills keep developing, which would make any glaring grammatical errors less apparent with time. The [ENNI](ENNI) and [Gillam](http://childes.psy.cmu.edu/access/Clinical-MOR/Gillam.html) Corpora also available on CHILDES have a combined 1147 children from a much younger age range of 4 to 11 years old. It just made sense to try recreate Gabani with better data.


{% include image.html
           img="assets/mlu_age.png"
           title="mlu_morphemes by age"
           caption="Figure 1: Mean Length of Utterances (MLU) in Morphemes (a measure of linguistic productivity) increases with age for both Specific Language Impaired (SLI) and Typically Developing (TD) children over all three corpora" %}

Secondly, there were several more features available simply through the KIDEVAL process on the [CLAN](http://talkbank.org/clan/) program provided with CHILDES that Gabani failed to extract. I of course thought of some more but I wanted to see if pumping more features into the machine learning algorithms would help.

Lastly, they trained their [part-of-speech tagger](https://en.wikipedia.org/wiki/Part-of-speech_tagging) on the Switchboard Corpus. I thought it was a little abstract since these narratives typically were one sided conversations. The [Brown Corpus](https://en.wikipedia.org/wiki/Brown_Corpus) offers a much more thorough tagset in regards to verbs thus it might be better at classifying the grammar of each word correctly.

## Using the features to gain some insights into the nature of SLI
As 



[ENNI]: https://www.researchgate.net/publication/230662487_Storytelling_from_pictures_using_the_Edmonton_Narrative_Norms_Instrument
