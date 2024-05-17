import re, os, random, collections, gc, requests
import time
from datetime import datetime

import Bio.PDB.PDBParser

import gensim.models
import pandas
import pandas as pd
import prompt_toolkit.key_binding.bindings.search
import seaborn
import taxonomyDict
import matplotlib.pyplot as plt
from urllib.request import urlretrieve
import numpy as np
from Bio import SwissProt, Entrez
from Bio import SeqIO, SeqRecord, Seq
from gensim import utils,models
from gensim.models.callbacks import CallbackAny2Vec
Entrez.email="shmotyamax@gmail.com"
import logging
import pickle
from tqdm import tqdm
import joblib
import more_itertools



random.seed(123)
plt.rcParams.update({'font.size': 18})


def toxProtParseUniProt(file):
    """Parse UniProt file, return list of SeqRecords"""
    toxProtFile=open(file)
    toxProt=[record for record in SwissProt.parse(toxProtFile)]
    return toxProt

def toxProtLengthsHistogram(records,nbins=25,range=(1,200),taxadict=None):
    """Draw distribution of length from UniProt records"""
    if taxadict==None:
        lengths=[]
        for prot in records:
            if prot.features:
                for feat in prot.features:
                    if feat.type=="CHAIN" or feat.type=="PEPTIDE":
                        try:
                            lengths.append(feat.location.end-feat.location.start)
                        except: continue
            else:lengths.append(len(prot.seq))
        plt.hist(lengths, bins=nbins, range=range)
        plt.xlabel("Довжина послідовності")
        plt.ylabel("Кількість")
        plt.show()
    else:
        lengths = {}
        for value in taxadict.values():
            lengths[value]=[]
        for prot in records:
            if prot.features:
                for feat in prot.features:
                    if feat.type == "CHAIN" or feat.type == "PEPTIDE":
                        try:
                            lengths[taxadict[prot.taxonomy_id[0]]].append((feat.location.end - feat.location.start))
                        except:
                            continue
            else:
                lengths[taxadict[prot.taxonomy_id[0]]].append(len(prot.seq))

        for key in list(lengths.keys())[:7]:
            plt.hist(lengths[key], bins=nbins, range=range)
            plt.title(key.capitalize())
            plt.show()
        other=[]

        for key in list(lengths.keys())[7:]:
            other+=lengths[key]
        lengths["other"] = other
        plt.hist(other, bins=nbins, range=range)
        plt.title("Other")
        plt.show()

def findKeyword(keywords,string):
    """Find keyword from list in string"""
    for word in keywords:
        if string.find(word)!=-1:
            return word
    return "Other"

def tpHistogramPandas(records, taxadict=None, nbins=25, range=(1,200), nontox_dict={}, tox=True):
    """Draw distribution of UniProt records by keywords or taxa"""
    pddict={"feat_id":[],"taxonomy": [], "feat_len": [], "keyword":[], "nontox":[]}
    for prot in records:
        if prot.features:
            for feat in prot.features:
                if feat.type == "CHAIN" or feat.type == "PEPTIDE":
                    try:
                        feat_len=feat.location.end - feat.location.start
                    except: continue
                    feat_id=prot.accessions
                    for idd in feat_id:
                        if idd in nontox_dict.keys():
                            pddict["nontox"].append(True)
                            break
                    else: pddict["nontox"].append(False)
                    if taxadict:
                        taxonomy=taxadict[prot.taxonomy_id[0]]
                        pddict["taxonomy"].append(taxonomy)
                    else:
                        pddict["taxonomy"].append(None)
                    pddict["feat_id"].append(feat_id)
                    pddict["feat_len"].append(feat_len)
                    pddict["keyword"].append(",".join(prot.keywords))


        else:
            feat_id = prot.accessions
            for idd in feat_id:
                if idd in nontox_dict.keys():
                    pddict["nontox"].append(True)
                    break
            else:
                pddict["nontox"].append(False)
            if taxadict:
                taxonomy = taxadict[prot.taxonomy_id[0]]
                pddict["taxonomy"].append(taxonomy)
            else:
                pddict["taxonomy"].append(None)
            pddict["feat_len"].append(len(prot.seq))
            pddict["keyword"].append(",".join(prot.keywords))

    df=pandas.DataFrame(pddict)

    if taxadict:
        taxcounts=collections.Counter(df.taxonomy)
        print(taxcounts)
        taxcount=taxcounts.most_common(7)
        taxlist={taxa for taxa,n in taxcount}
        taxcountss=set(taxcounts.keys())
        tax_to_replace=taxcountss.difference(taxlist)
        df.sort_values(by=["taxonomy"], key=lambda x: x.apply(lambda y: taxcounts[y]), inplace=True, ascending=False)
        df.replace(to_replace=tax_to_replace,value="other",inplace=True)
        plot = seaborn.histplot(df, x="feat_len", hue="taxonomy", multiple="stack", bins=nbins, binrange=range)
        plot.set(xlabel="Довжина послідовності", ylabel="Кількість")
        plt.show()
    plot=seaborn.histplot(df,x="feat_len",hue="nontox",multiple="stack",bins=nbins,binrange=range)
    plot.set(xlabel="Довжина послідовності", ylabel="Кількість")
    plt.show()
    keywords=list(df.keyword)
    keywordsCount={}
    for instance in keywords:
        for item in instance.split(","):
            keywordsCount[item]=keywordsCount.get(item,0)+1
    print(keywordsCount)
    if tox:
        toxkeywords=["Neurotoxin","Hemostasis impairing toxin","Antimicrobial","Cytolysis","G-protein coupled receptor impairing toxin","Hypotensive agent","Protease inhibitor","Ion channel impairing toxin"]
        df["keyword"]=df["keyword"].map(lambda x: findKeyword(toxkeywords,x))
    else:
        keywords=["Antimicrobial", "Nematocyst", "Secreted"]
        df["keyword"]=df["keyword"].map(lambda x: findKeyword(keywords,x))
    keywords=list(df.keyword)
    keywordsCount={}
    for instance in keywords:
        for item in instance.split(","):
            keywordsCount[item]=keywordsCount.get(item,0)+1
    print(keywordsCount)
    df.sort_values(by=["keyword"], key=lambda x: x.apply(lambda y: keywordsCount[y]), inplace=True, ascending=False)
    plot = seaborn.histplot(df, x="feat_len", hue="keyword", multiple="stack", bins=nbins, binrange=range)
    plot.set(xlabel="Довжина послідовності", ylabel="Кількість")
    plt.show()

def writeSeqsToFasta(records,filename,cut=False,indexes=None):
    """Write UniProt records to fasta"""
    seqs=[]
    fullseqs=[]
    length=0
    cuts=0

    for prot in records:
        if indexes == None:
            index=str(len(fullseqs))
        else:
            index=indexes[prot.accessions[0]]
        seqs.append(SeqRecord.SeqRecord(Seq.Seq(prot.sequence),id=prot.accessions[0],description=str(prot.entry_name)+" "+str(index)))
        fullseqs.append(SeqRecord.SeqRecord(Seq.Seq(prot.sequence),id=prot.accessions[0],description=str(prot.entry_name)+" "+str(index)))
        length+=prot.sequence_length
        if cut == True and length>99000:
            length=0
            SeqIO.write(seqs, filename + f"_{cuts}.fa", "fasta")

            seqs=[]
            cuts+=1
    SeqIO.write(fullseqs, filename + ".fa", "fasta")

def randomizedSubset(records,percent,filename):
    """Subset sequences"""
    randomseqs=[]
    for prot in records:
        if random.randint(0,100)<percent and 20<len(prot.seq)<150: randomseqs.append(SeqRecord.SeqRecord(Seq.Seq(prot.seq),id=prot.id))
    SeqIO.write(randomseqs,filename,"fasta")

def ParseXMLforAccesion(filename):
    """Parses BLAST output xml for accessions IDs"""
    XML=open(filename)
    newXML=XML.read()
    reference_re=re.compile("query-title>(.+)<")
    accessions_re=re.compile("accession>(.+)<")
    reference_tox=re.search(reference_re,newXML).groups()[0].split()[0]
    accesions=re.findall(accessions_re,newXML)
    return accesions,reference_tox

def accessionInToxProt(accession,toxprot):
    """Check if accession ID is in ToxProt"""
    for acc in accession:
        if acc in toxprot:
            return True
    return False

def retrieveAccessionFromUniprot(accession, folder="Uniprot_entries"):
    """Download UniProt file for protein with accession"""
    urlretrieve(f"https://rest.uniprot.org/uniprotkb/{accession}?format=txt", f"{folder}/{accession}.txt")
    time.sleep(0.01)

def analyzeBlast(Blastdir, unrevtoxprot, cache=None):
    """Analyze all web BLAST output files in xml format.\n
    Finds ToxProt sequences, which have non-toxic homologs"""
    files = os.listdir(Blastdir)
    cur=0
    fullacc = []
    newrecords=[]
    refacc={}
    NAUniprot=[]
    accs=toxProtAllAccs(unrevtoxprot)
    revrefacc={}
    if cache:
        with open(cache) as file:
            for line in file:
                if line.find(":")==-1:
                    fullacc.append(line.strip())
                else:
                    revrefacc[line.strip().split(":")[0]]=re.findall("'(.*?)'",line)
            refacc=reverseDict(revrefacc)
    else:
        for file in files:
            accessions,reference_tox = ParseXMLforAccesion(f"Blast_vs_UniProt/{file}")
            for id in accessions:
                b=accessionInToxProt(id, accs)
                if id not in fullacc and not b:
                    fullacc.append(id)
                    refacc[id]=[reference_tox]
                elif not b and reference_tox not in refacc[id]:
                    refacc[id].append(reference_tox)
            cur+=1
            print(cur/len(files),reference_tox)
        revrefacc = reverseDict(refacc)
        with open(f"cache_{str(datetime.now())[:10].replace('-','_')}","w") as cache_out:
            cache_out.write("\n".join(fullacc))
            cache_out.writelines([f"\n{id}:{revrefacc[id]}" for id in revrefacc])
    entries = os.listdir("Uniprot_entries")
    newUniAccs=[]
    keywords={}
    for acc in fullacc:
        if acc + ".txt" not in entries:
            retrieveAccessionFromUniprot(acc)
            newUniAccs.append(acc)
        try:
            record = SwissProt.read(f"Uniprot_entries/{acc}.txt")
        except:
            NAUniprot.append(acc)
            continue
        if "Toxin" in record.keywords:
            refacc.pop(acc)
            continue
        if record.features:
            for feat in record.features:
                if feat.type == "CHAIN" or feat.type == "PEPTIDE":
                    try:
                        if (int(feat.location.end) - int(feat.location.start)) < 150 and record.sequence_length<300:
                            newrecords.append(SwissProt.Record())
                            newrecords[-1].sequence=record.sequence[int(feat.location.start):int(feat.location.end)]
                            newrecords[-1].sequence_length=int(feat.location.end) - int(feat.location.start)
                            newrecords[-1].accessions=record.accessions.copy()
                            newrecords[-1].entry_name=record.entry_name
                    except:
                        if record.sequence_length < 150:
                            newrecords.append(record)
                            break
        elif record.sequence_length < 150:
            newrecords.append(record)
        else:
            refacc.pop(acc)
            continue
        for keyword in record.keywords:
            keywords[keyword]=keywords.get(keyword,0)+1
    print("No UniProt accessions:", NAUniprot)
    print(f"Downloaded {len(newUniAccs)} new UniProt accessions")
    writeSeqsToFasta(newrecords, "nt_to_clust_last_f")
    return refacc, keywords


def toxProtAllAccs(toxprot):
    """Find all accession IDs in ToxProt list and return as list"""
    accs=[]
    for _ in toxprot:
        accs+=_.accessions
    return accs

def read_fasta(file):
    """Parse fasta file, return list of SeqRecords"""
    seqlist=[]
    with open(file) as hand:
        for seq in SeqIO.parse(hand,"fasta"):
            seqlist.append(seq)
        return seqlist


def reverseDict(dicti):
    """Reverse dictionary keys and values"""
    reversedDict={}
    for key in dicti:
        for item in dicti[key]:
            if item not in reversedDict.keys():
                reversedDict[item]=[]
            reversedDict[item].append(key)
    return reversedDict

def prepareToBlast(records,lenmax=150,lenmin=10,cut=True,filename="records_to_blast",shuffle=False,numshuffles=5):
    """Select and write UniProt records of suitable length to fasta, with option to cut file into smaller files, which can be BLASTed online"""
    newrecords=[]
    for record in records:
        if record.features:
            for feat in record.features:
                if feat.type == "CHAIN" or feat.type == "PEPTIDE":
                    try:
                        if lenmin < (int(feat.location.end) - int(feat.location.start)) < lenmax:
                            newrecords.append(SwissProt.Record())
                            newrecords[-1].sequence=record.sequence[int(feat.location.start):int(feat.location.end)]
                            newrecords[-1].sequence_length=int(feat.location.end) - int(feat.location.start)
                            newrecords[-1].accessions=record.accessions.copy()
                            newrecords[-1].entry_name=record.entry_name
                            newrecords[-1].features=[feat]
                    except:
                        if record.sequence_length < lenmax:
                            newrecords.append(record)
                            break
        elif record.sequence_length < lenmax:
            newrecords.append(record)
    if shuffle:
        shuffled=[]
        for record in newrecords:
            for num in range(numshuffles):
                shuffled.append(SwissProt.Record())
                shuffled[-1].sequence_length=record.sequence_length
                shuffled[-1].accessions = record.accessions.copy()
                shuffled[-1].entry_name = record.entry_name+"_shuffled"
                shuffled[-1].features = record.features.copy()
                shuffled_seq=list(record.sequence)
                random.shuffle(shuffled_seq)
                shuffled[-1].sequence = "".join(shuffled_seq)
        writeSeqsToFasta(shuffled, filename, cut=cut)
    else:
        writeSeqsToFasta(newrecords,filename,cut=cut)

def retrievePDB(PDB: str):
    url = f"https://alphafold.ebi.ac.uk/api/prediction/{PDB}"
    try:
        res = requests.get(url).json()[0]
    except:
        return
    print(res)
    urlretrieve(res["pdbUrl"], f"False_classified/PDB/{PDB}.pdb")

def createBashRF(dir: str):
    flist = os.listdir(dir)
    with open("anotherRFbash.sh", "w") as sh:
        lines=["#!/bin/bash"]
        for file in flist:
            parser = Bio.PDB.PDBParser()
            structure = parser.get_structure(file[:-3], f"{dir}/{file}")

            #chains = [f"{len(chain)}-{len(chain)}/0 " for chain in structure.get_chains()]
            #contigs="".join(chains)[:-3]
            model = structure[0]
            residue_to_remove = []
            chain_to_remove = []
            for chain in model:
                for residue in chain:
                    if residue.id[0] != ' ':
                        residue_to_remove.append((chain.id, residue.id))
                if len(chain) == 0:
                    chain_to_remove.append(chain.id)

            for residue in residue_to_remove:
                model[residue[0]].detach_child(residue[1])

            for chain in chain_to_remove:
                model.detach_child(chain)

            for to_save in model:
                x = Bio.PDB.PDBIO()
                x.set_structure(to_save)
                x.save(f"False_Classified_FP/chains/{file}")
                contigs=f"{len(to_save)}-{len(to_save)}"
                break
            lines.append(f"\ndocker run -it --rm   -v $HOME/models:$HOME/models   -v $HOME/inputs:$HOME/inputs "
                         f"-v $HOME/outputs:$HOME/outputs   rfdiffusion   "
                         f"inference.output_prefix=$HOME/outputs/False_classified_RF/{file[:-4]} "
                         f"inference.model_directory_path=$HOME/models inference.input_pdb=$HOME/inputs/{file} "
                         f"inference.num_designs=3 denoiser.noise_scale_ca=0.5 denoiser.noise_scale_frame=0.5 "
                         f"'contigmap.contigs=[{contigs}]' diffuser.partial_T=10")

        sh.writelines(lines)

def ExtractSeqFromPDB(dir):
    d3to1 = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
             'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
             'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
             'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}
    flist = os.listdir(dir)
    with open("diffseqs.fa", "a") as seqs:
        for file in flist:
            parser = Bio.PDB.MMCIFParser()
            structure = parser.get_structure(file[:-4], f"{dir}/{file}")
            seq = "".join([d3to1[r.resname] for r in structure[0]["A"].get_residues()])
            seqs.write(f">{file[:-4]}\n{seq}\n")

def extractSeqFromPsychopathCSV(file) -> list[str]:
    with open(file, "r") as csv:
        records = []
        for line in csv:
            records.append(line.strip())
    return records

def slidingWindow(sequence,wlen=3):
    '''Create list of triplets from sequence'''
    sentence=[]
    for i in range((len(sequence)-wlen+1)):
        sentence.append(str(sequence[i:i+wlen]))
    return sentence

def createCorpus(testPerc=25):
    '''Create 2 lists of TaggedDocuments:
    1) Training set (corpus) for model training
    2) Testing set (testCorpus) for model assession'''
    corpus = []
    testCorpus = []
    print("Creating corpus")
    try:
        with open("uniprot_slide_2.py", "r") as uniDictTxt:
            for line in uniDictTxt:
                line = str(line).split(":")
                if random.randint(1, 100) <= (100 - testPerc):
                    corpus.append(models.doc2vec.TaggedDocument(re.findall("'(.{3})'", line[1].upper()),
                                                                [re.search("'(.*)'", line[0])[1]]))
                else:
                    testCorpus.append(models.doc2vec.TaggedDocument(re.findall("'(.{3})'", line[1].upper()),
                                                                    [re.search("'(.*)'", line[0])[1]]))
    except:
        with open("uniprot_sprot_nr.fasta") as fh:
            for record in SeqIO.parse(fh, "fasta"):
                doc = models.doc2vec.TaggedDocument(words=slidingWindow(record.seq),
                                                    tags=[re.search("\|(.*)\|", record.id)])
                if random.randint(1, 100) <= (100 - testPerc):
                    corpus.append(doc)
                else:
                    testCorpus.append(doc)
    return corpus, testCorpus

def createToxCorpus(only_tags=False):
    '''Create list of TaggedDocuments containing only toxic sequences (corpus)\n
    And list of their UniProt accession IDs/names (taglist)'''
    corpus=[]
    taglist=[]
    print("Creating tox. corpus")
    file="uniprot_toxprot_rev.fasta"
    with open(file,"r") as toxprot:
        for record in SeqIO.parse(toxprot,"fasta"):
            if only_tags==False:
                corpus.append(models.doc2vec.TaggedDocument(slidingWindow(record.seq),
                                                            [f"sp|{record.description.split()[0]}|{record.description.split()[1]}"]))
            taglist.append(f"sp|{record.description.split()[0]}|{record.description.split()[1]}")
            taglist.append(record.description.split()[0])
    return corpus, taglist


def trainEmbedding(corpus, nepochs=25, ndims=100, min_count=3, dm=0, neg=5, callbacks=()):
    '''Train model on corpus with chosen parameters'''
    model=models.doc2vec.Doc2Vec(vector_size=ndims,min_count=min_count,epochs=nepochs,dm=dm,negative=neg)
    model.build_vocab(corpus)
    model.train(corpus,total_examples=model.corpus_count,epochs=model.epochs,callbacks=callbacks)
    return model

def corpusFindSimilars(model,corpus,topn=10):
    """Infer vectors from corpus using pre-trained model and find tags with the most similar vectors"""
    similars={}
    print("Vectorizing...")
    for doc in tqdm(corpus):
        inferred_vector = model.infer_vector(doc.words)
        similars[doc.tags[0]]=[tag for tag, sim in model.dv.most_similar(inferred_vector, topn=topn) if tag!=doc.tags[0]]
    return similars

def assessModel(model,corpus):
    '''Assess model performance:
    Uses training set as test set\n
    Shows if any sequence is found to be the most similar to itself (good, 0-4) or other sequences (bad, 5+)'''
    ranks = []
    n=0
    for doc_id in tqdm(corpus):
        if True:
            inferred_vector = model.infer_vector(doc_id.words)
            sims = model.dv.most_similar(inferred_vector, topn=5)
            try:
                rank = [docid for docid, sim in sims].index(doc_id.tags[0])
            except ValueError:
                rank = "5+"
            ranks.append(rank)
        n+=1
    count=collections.Counter(ranks)
    print(count)
    return count

def alternativeBlastAssession(model,testCorpus,topn=15,top_hits=15,top_perc=None):
    '''Assess model performance on testCorpus in comparison to BLAST\n
    Assesses how good is model at embedding sequence similarity\n
    10 - full alignment between BLAST and the model,
    0 - no alignment'''
    similars = {}
    assession=[]
    for prot in tqdm(testCorpus):
        vec = model.infer_vector(prot.words)
        similars[prot.tags[0]] = [tag for tag, sim in model.dv.most_similar(vec, topn=topn)]
    for testSeq in tqdm(similars):
        accname=testSeq.split("|")[1]
        try:
            blast_res=pd.read_csv(f"Blast_splitted/{accname}.csv", names=["qname", "qlen", "sname", "slen", "eval", "hlen"], index_col=0)
        except:
            assession.append("Empty")
            continue
        blast_res.sort_values(by=["eval"],inplace=True)
        blast_res.drop_duplicates()
        if top_perc:
            top_hits_prot=max((min(top_hits,len(blast_res.index)),int(len(blast_res.index)*top_perc)))
        else: top_hits_prot=min(top_hits,len(blast_res.index))
        blast_res_top=set(blast_res.iloc[:top_hits_prot,1])
        intersection=list(blast_res_top & set(similars[testSeq]))
        assession.append(len(intersection))
    counter=collections.Counter(assession)
    print(counter)
    plotx=[(x,counter[x]) for x in counter if x!="Empty"]
    plotx.sort()
    plt.plot(plotx[0],plotx[1])
    plt.show()
    return counter

def votingAssession(model, testcorpus, toxprot, topn=5):
    '''Assess model ability to distinguish toxic and non-toxic proteins.
    Positive = toxic, Negative = non-toxic'''
    similars={}
    predictionsn=[]
    predictions1=[]
    reals=[]
    resultsn=[]
    results1 = []
    for prot in tqdm(testcorpus):
        if prot.tags[0] in toxprot:
            reals.append("T")
        else:
            reals.append("F")
        vec = model.infer_vector(prot.words)
        similars[prot.tags[0]] = [tag for tag, sim in model.dv.most_similar(vec, topn=topn+1) if tag!=prot.tags[0]]
        for i,tag in enumerate(similars[prot.tags[0]]):
            if i==0:
                if tag in toxprot:
                    predictions1.append("T")
                else:
                    predictions1.append("F")
            elif tag in toxprot and i<topn:
                similars[prot.tags[0]][i]="tox"
            elif i<topn:
                similars[prot.tags[0]][i]="nontox"
        toxcount=collections.Counter(similars[prot.tags[0]])
        if toxcount["tox"]>toxcount["nontox"]:
            predictionsn.append("T")
        else: predictionsn.append("F")
    print(len(predictionsn))
    print(len(predictions1))
    print(len(reals))
    for p,r in zip(predictionsn,reals):
        if p==r=="T":
            resultsn.append("TP")
        elif p==r=="F":
            resultsn.append("TN")
        elif p=="T" and r=="F":
            resultsn.append("FP")
        elif p=="F" and r=="T":
            resultsn.append("FN")
    for p,r in zip(predictions1,reals):
        if p==r=="T":
            results1.append("TP")
        elif p==r=="F":
            results1.append("TN")
        elif p=="T" and r=="F":
            results1.append("FP")
        elif p=="F" and r=="T":
            results1.append("FN")
    rescount1=collections.Counter(results1)
    rescountn = collections.Counter(resultsn)
    print(f"Number of predictions with voting: {rescountn}\n"
          f"Accuracy: {(rescountn['TP']+rescountn['TN'])/len(resultsn)}")
    print(f"Number of predictions with single best hit: {rescount1}\n"
          f"Accuracy: {(rescount1['TP'] + rescount1['TN']) / len(results1)}")

def assessExistingModel(model, testCorpus, topn=15, top_hits=15, perc=None, voting=False, votingtaglist=None):
    '''Convenient way to import existing model and assess it'''
    mod=models.Doc2Vec.load(model)
    if voting==False:
        count=alternativeBlastAssession(mod, testCorpus, topn=topn, top_hits=top_hits, top_perc=perc)
        return count
    else:
        votingAssession(mod,testcorpus=testCorpus,toxprot=votingtaglist,topn=topn)

def blastVoting(toxlist,dir,topn=5):
    '''Assess BLAST ability to distinguish toxic and non-toxic proteins.
    Positive (P) = toxic, Negative (N) = non-toxic'''
    blastfiles=os.listdir(dir)
    reals = []
    predictions=[]
    results=[]
    nonblasted=[]
    for file in tqdm(blastfiles):
        cur=pd.read_csv(f"{dir}/{file}",names=["qname", "qlen", "sname", "slen", "eval", "hlen"], index_col=0)
        cur.sort_values(["eval","hlen"],inplace=True)
        cur.drop(columns=["eval","hlen"],inplace=True)
        cur.drop_duplicates(inplace=True)
        if cur.empty:
            results.append("None")
            nonblasted.append(file[:-4])
            continue
        if cur.index[0] in toxlist:
            reals.append("T")
        else:
            reals.append("F")
        inpred=[]
        size=min(len(cur.index),topn)
        for prot in list(cur.iloc[0:size,1]):
            if prot in toxlist:
                inpred.append("tox")
            else:
                inpred.append("nontox")
        toxcount = collections.Counter(inpred)
        if toxcount["tox"] > toxcount["nontox"]:
            predictions.append("T")
        else:
            predictions.append("F")
    print(len(predictions))
    print(len(reals))
    for p, r in zip(predictions, reals):
        if p == r == "T":
            results.append("TP")
        elif p == r == "F":
            results.append("TN")
        elif p == "T" and r == "F":
            results.append("FP")
        elif p == "F" and r == "T":
            results.append("FN")
    rescount = collections.Counter(results)
    print(f"Number of predictions: {rescount}\n"
          f"Accuracy: {(rescount['TP'] + rescount['TN']) / len(results)}")
    return nonblasted

def functionPrediction(model, testCorpus, topn=5):
    """Assess model ability to predict toxin function"""
    similars=corpusFindSimilars(model,testCorpus,topn=topn)
    uniprot=toxProtParseUniProt("uniprot_toxprot_rev.txt")
    functionset={"Cardiotoxin","Cell adhesion impairing toxin","Complement system impairing toxin","Dermonecrotic toxin",
                  "Enterotoxin","G-protein coupled receptor impairing toxin","Hemostasis impairing toxin",
                  "Ion channel impairing toxin","Myotoxin","Neurotoxin"}
    tag_to_kwlist={}
    results=[]
    betterresults=[]
    for record in uniprot:
        kwset=set(record.keywords)
        tag_to_kwlist[f"sp|{record.accessions[0]}|{record.entry_name}"]=list(kwset.intersection(functionset))
    for prot in similars:
        reals = set(tag_to_kwlist[prot])
        kwlist=[]
        for tag in similars[prot]:
            kwlist+=tag_to_kwlist.get(tag,[])
        predictor=set(collections.Counter(kwlist).keys())
        #
        betterkw=collections.Counter(kwlist)
        try:
            betterpredictor = sum([betterkw.get(val, 0) for val in reals]) / len(kwlist)
        except ZeroDivisionError:
            betterpredictor = 0
        if betterpredictor>0.75:
            betterresults.append("Accurate")
        elif betterpredictor>0.5 or (len(kwlist)<3 and len(reals)==0) or len(reals)==len(predictor)==0:
            betterresults.append("Plausible")
        elif betterpredictor>0.25:
            betterresults.append("Bad")
        else:
            betterresults.append("Inaccurate")
        #
        if len(reals)==len(predictor)==len(predictor.intersection(reals)) and len(reals)!=0:
            results.append("Accurate")
        elif (len(predictor.intersection(reals))>0 and len(predictor)+len(reals)<len(predictor.intersection(reals))*4) or len(reals) == len(predictor) == 0:
            results.append("Plausible")
        else:
            results.append("Completely inaccurate")
    print(f"Model {model.vector_size} 1: {collections.Counter(results)}")
    print(f"Model {model.vector_size} 2: {collections.Counter(betterresults)}")

def functionBlastPrediction(dir,topn=5):
    """Assess BLAST ability to predict toxin function"""
    uniprot = toxProtParseUniProt("uniprot_toxprot_rev.txt")
    functionset={"Cardiotoxin","Cell adhesion impairing toxin","Complement system impairing toxin","Dermonecrotic toxin",
                  "Enterotoxin","G-protein coupled receptor impairing toxin","Hemostasis impairing toxin",
                  "Ion channel impairing toxin","Myotoxin","Neurotoxin"}
    tag_to_kwlist={}
    results=[]
    betterresults=[]
    for record in uniprot:
        kwset=set(record.keywords)
        tag_to_kwlist[f"sp|{record.accessions[0]}|{record.entry_name}"]=list(kwset.intersection(functionset))
        tag_to_kwlist[record.accessions[0]] = list(kwset.intersection(functionset))
    blastfiles = os.listdir(dir)
    for file in tqdm(blastfiles):
        cur=pd.read_csv(f"{dir}/{file}",names=["qname", "qlen", "sname", "slen", "eval", "hlen"], index_col=0)
        cur.sort_values(["eval","hlen"],inplace=True)
        cur.drop(columns=["eval","hlen"],inplace=True)
        cur.drop_duplicates(inplace=True)
        if cur.empty:
            results.append("None")
            continue
        reals=set(tag_to_kwlist[cur.index[0]])
        preds=[]
        for prot in list(cur.iloc[0:topn, 1]):
            preds+=tag_to_kwlist.get(prot,[])
        predictor = set(collections.Counter(preds).keys())
        #
        betterkw=collections.Counter(preds)
        try:
           betterpredictor=sum([betterkw.get(val,0) for val in reals])/len(preds)
        except ZeroDivisionError:
            betterpredictor=0
        if betterpredictor>0.75:
            betterresults.append("Accurate")
        elif betterpredictor>0.5 or (len(preds)<3 and len(reals)==0) or len(reals)==len(predictor)==0:
            betterresults.append("Plausible")
        elif betterpredictor>0.25:
            betterresults.append("Bad")
        else:
            betterresults.append("Inaccurate")
        #
        if len(reals) == len(predictor) == len(predictor.intersection(reals)) and len(reals)!=0:
            results.append("Accurate")
        elif (len(predictor.intersection(reals)) > 0 and len(predictor)+len(reals)<len(predictor.intersection(reals))*4) or len(reals) == len(predictor) == 0:
            results.append("Plausible")
        else:
            results.append("Completely inaccurate")
    print(f"BLAST 1: {collections.Counter(results)}")
    print(f"BLAST 2: {collections.Counter(betterresults)}")

def embedding(model: str|models.Doc2Vec, corpus: list, for_training=True, outer_taglist=None):
    """Convert corpus sequences to vectors"""
    if type(model) == str:
        mod = models.Doc2Vec.load(model)
    else:
        mod = model
    if outer_taglist:
        taglist=outer_taglist
    else:
        _,taglist=createToxCorpus(only_tags=True)
        del _
        gc.collect()
    vectorized = [[], [], []]
    for prot in tqdm(corpus):
        vector=mod.infer_vector(prot.words)
        vectorized[0].append(vector)
        vectorized[2].append(prot.tags[0])
        if for_training:
            if prot.tags[0] in taglist:
                vectorized[1].append(1)
            else:
                vectorized[1].append(0)
    return vectorized

def multiprocess_embedding(model: str, corpus: list, for_training=True,
                           outer_taglist=None, n_threads=4, mulitags=False):
    """Convert corpus sequences to vectors using multiprocessing (faster)"""
    if not mulitags:
        if not outer_taglist:
            _, taglist = createToxCorpus(only_tags=True)
        else:
            taglist=outer_taglist
        mod = models.Doc2Vec.load(model)
        corpus = [list(c) for c in more_itertools.divide(n_threads, corpus)]
        result = joblib.Parallel(n_jobs=n_threads)(
        joblib.delayed(embedding)(mod, doc, for_training=for_training, outer_taglist=taglist) for doc in corpus)
    else:
        """
        keywords = {"Multifunctional enzyme", "Activator", "Chaperone", "Developmental protein", "DNA-binding", "Hydrolase",
                 "Isomerase", "Ligase",
                         "Lyase", "Oxidoreductase", "Receptor", "Ribonucleoprotein", "RNA-binding",
                         "Transducer", "Transferase", "Translocase"}
        tags={}
        for prot in tqdm(SwissProt.parse("uniprot_sprot.dat")):
            if prot.keywords:
                if "Toxin" in prot.keywords:
                    tags[prot.accessions[0]]="Toxin"
                elif len(its := keywords.intersection(set(prot.keywords)))>0:
                    tags[prot.accessions[0]]=list(its)[0]
                else:
                    tags[prot.accessions[0]] = "Other"
            else:
                tags[prot.accessions[0]] = "Other"
        corpus = [list(c) for c in more_itertools.divide(n_threads, corpus)]
        result = joblib.Parallel(n_jobs=n_threads)(joblib.delayed(embeddingWithKeywords)(model, doc, tags) for doc in corpus)"""
        raise Exception("'multitags' parameter is disabled")

    del corpus
    gc.collect()
    complete = [[], [], []]
    for llist in result:
        complete[0] += llist[0]
        complete[1] += llist[1]
        complete[2] += llist[2]
    return complete

class ModelSaverAssessor(CallbackAny2Vec):
    def __init__(self,corpus,name):
        self.epochs=[]
        self.corpus=corpus
        self.assessments=[]
        self.nep=1
        self.models=[]
        self.name=name
    def on_epoch_end(self, model):
        if self.nep%5==0:
            model.save(f"TEST_WITH_CLUSTERING/tempnep{self.name}_{self.nep}")
            #self.models.append(models.Doc2Vec.load(f"neps/tempnep{self.name}"))
            #self.epochs.append(self.nep)
        self.nep+=1
    #def on_train_end(self, model):
    #    for mod in self.models:
    #        count = assessModel(mod,self.corpus)
    #        try:
    #            self.assessments.append((len(self.corpus)-count["Empty"]-count[0])/(len(self.corpus)-count["Empty"]))
    #        except:
    #            self.assessments.append(0)
    #    plt.plot(self.epochs,self.assessments)
    #    plt.xlabel("Epoch")
    #    plt.ylabel("Accuracy")
    #    plt.title(f"Model training assessment ({model.vector_size} dimensions)")
    #    plt.show()

def test():
    #logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    """ model = models.Doc2Vec.load("production_model")
    corpus, testCorpus = createCorpus(test=True, testPerc=26)
    assessModel(model, testCorpus)
    model = models.Doc2Vec.load("production_model_100")
    assessModel(model, testCorpus)"""
    """
    uniprot = toxProtParseUniProt("uniprot_sprot.dat")
    newrecords = []
    residues_to_skip = ["X", "B", "Z", "U", "J", "O"]
    for r in uniprot:
        if any([name in r.sequence for name in residues_to_skip]) or "Toxin" in r.keywords:
            continue
        else:
            if random.randint(1,100) == 1:
                newrecords.append(r)
        if len(newrecords) == 6741:
             break
    writeSeqsToFasta(newrecords, "ToxPredFastaNonTox.fa")"""


    """negs = extractSeqFromPsychopathCSV("ToxPred3test_neg.csv")
    poss = extractSeqFromPsychopathCSV("ToxPred3test_pos.csv")
    ncorpus = []
    pcorpus = []
    for i, n in enumerate(negs):
        ncorpus.append(models.doc2vec.TaggedDocument(slidingWindow(n), [str(i)]))
    for i, p in enumerate(poss):
        pcorpus.append(models.doc2vec.TaggedDocument(slidingWindow(p), [str(i)]))
    _, taglist = createToxCorpus(only_tags=True, rev=True, new=False)
    nembeded = multiprocess_embedding("production_model_100", ncorpus, for_training=True, outer_taglist=taglist)
    nembeded[1] = np.zeros(len(nembeded[1]))
    pembeded = multiprocess_embedding("production_model_100", pcorpus, for_training=True, outer_taglist=taglist)
    pembeded[1] = np.ones(len(pembeded[1]))
    with open("ToxPredN_100.pickle","wb") as ut:
        pickle.dump(nembeded, ut)
    with open("ToxPredP_100.pickle","wb") as ut:
        pickle.dump(pembeded, ut)"""
    """records = toxProtParseUniProt("uniprot_toxprot_rev.txt")
    newrecords = []
    residues_to_skip = ["X", "B", "Z", "U", "J", "O"]
    for r in records:
        if any([name in r.sequence for name in residues_to_skip]):
            continue
        else:
            newrecords.append(r)
    writeSeqsToFasta(newrecords, "ToxPred/ToxpredFasta", cut=True)"""



    #ExtractSeqFromPDB("Chroma_diffused_tox")

    """toxprot = toxProtParseUniProt("uniprot_toxprot_rev.txt")
    PDBaccs = []
    for rec in toxprot:
        if 20 < rec.sequence_length < 150:
            for tup in rec.cross_references:
                if tup[0] == "PDB":
                    PDBaccs.append(tup[1]+"\n")
                    break
    with open("to_diffuse.txt", "w") as tdf:
        tdf.writelines(PDBaccs)"""
    #x = Bio.PDB.PDBList()
    #x.download_pdb_files(pdb_codes=PDBaccs, pdir="To_diffuse", file_format="pdb")

    """records=[]
    for seq in SeqIO.parse("diffseqs_filt.fasta", "fasta"):
        records.append(models.doc2vec.TaggedDocument(words=slidingWindow(seq.seq), tags=[seq.id]))
    _, taglist = createToxCorpus(only_tags=True, rev=True, new=False)
    embeded = multiprocess_embedding("production_model_100", records, for_training=True, outer_taglist=taglist)
    with open("diffseqs_100.pickle","wb") as ut:
        pickle.dump(embeded, ut)"""

    #createBashRF("False_classified_FP/PDB")
    #taxa = []
    """records = []
    for file in os.listdir("False_classified_FP"):
        if file.endswith(".txt"):
            records.append(SwissProt.read(f"False_classified_FP/{file}"))
    PDBaccs = []
    for rec in records:
        if 20 < rec.sequence_length < 150:
            for tup in rec.cross_references:
                if tup[0] == "PDB":
                    PDBaccs.append(tup[1])
    """
    """records = []
    for file in os.listdir("RF_MPNN/tox2"):
        with open("RF_MPNN/tox2/"+file, "r") as fh:
            for i, record in enumerate(SeqIO.parse(fh, "fasta")):
                if i == 0:
                    continue
                records.append(models.doc2vec.TaggedDocument(slidingWindow(record.seq), file+record.description[9:17]))
    _, taglist = createToxCorpus(only_tags=True, rev=True, new=False)
    embeded = multiprocess_embedding("production_model_100", records, for_training=True, outer_taglist=taglist)
    embeded[1] = np.ones(len(embeded[1]))
    with open("RFMPNN2_100.pickle", "wb") as p:
        pickle.dump(embeded, p)"""

    """
    
    """

    """toxprot = toxProtParseUniProt("uniprot_toxprot_rev.txt")
    reference_acc, _ = analyzeBlast("Blast_vs_UniProt", toxprot, cache="cache_2023_12_26")
    taxa = []
    records = toxprot"""
    """for file in os.listdir("temp"):
        if file.endswith(".txt"):
            records.append(SwissProt.read(f"False_classified_FP/{file}"))"""
    """for rec in records:
        if rec.taxonomy_id[0] not in taxa: taxa.append(rec.taxonomy_id[0])
    record = []
    recordDict = {}
    for _ in range(len(taxa) // 200 + 1):
        try:
            taxData = Entrez.esummary(db="taxonomy", id=taxa[_ * 200:_ * 200 + 200])
        except:
            taxData = Entrez.esummary(db="taxonomy", id=taxa[_ * 200:])
        record += Entrez.read(taxData)

    for _ in record:
        recordDict[_["Id"]] = _["Division"]
    print(recordDict)


    toxProtLengthsHistogram(records, nbins=50, range=(0, 250))
    tpHistogramPandas(toxprot, taxadict=recordDict, nbins=50, range=(0, 250), tox=True)"""

def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    corpus, testCorpus = createCorpus(testPerc=26)
    model=trainEmbedding(corpus, nepochs=50, ndims=100, neg=5, dm=0, min_count=5)
    model.save("production_model_100")
    tcorp, taglist = createToxCorpus(only_tags=False)
    embeded = multiprocess_embedding(model="production_model", corpus=corpus, for_training=True, outer_taglist=taglist, n_threads=8)
    with open("train_embedded.pickle","wb") as ut:
        pickle.dump(embeded, ut)
    embeded = multiprocess_embedding(model="production_model", corpus=testCorpus, for_training=True, outer_taglist=taglist, n_threads=8)
    with open("test_embedded.pickle","wb") as ut:
        pickle.dump(embeded, ut)


if __name__ == "__main__":
    #test()
    main()

# UNUSED
"""def preparingToBlastToxinFamilyMembers(keyword):
    toxprot=toxProtParseUniProt("uniprot_toxprot_rev.txt")
    members=[]
    for record in toxprot:
        if keyword in record.keywords:
            members.append(record)
    writeSeqsToFasta(members,f"full_of_{keyword}")

def createDBWithoutKeywordToxins():
    files=os.listdir("KeywordFamilies")
    homologs={}
    out_seqs={}
    for file in files:
        df=pd.read_csv(f"KeywordFamilies/{file}",names=["qname", "qlen", "sname", "slen", "idperc", "pperc", "hlen"], index_col=0)
        df=df.loc[df.idperc>30]
        homologs[file]=set(df.sname)
        out_seqs[file]=[]
        for record in SeqIO.parse("uniprot_sprot_nr.fasta","fasta"):
            if record.name not in homologs[file]:
                out_seqs[file].append(record)
        with open(f"Without_{file[:-4]}.fasta","w") as fhand:
            SeqIO.write(out_seqs[file],fhand,"fasta")"""

"""def embeddingWithKeywords(model: str|models.Doc2Vec, corpus: list, tags=None):
    if tags==None:
        keywords = {"Multifunctional enzyme", "Activator", "Chaperone", "Developmental protein", "DNA-binding", "Hydrolase",
                 "Isomerase", "Ligase",
                 "Lyase", "Oxidoreductase", "Receptor", "Ribonucleoprotein", "RNA-binding",
                 "Transducer", "Transferase", "Translocase"}
        tags={}
        for prot in SwissProt.parse("uniprot_sprot.dat"):
            if prot.keywords:
                if "Toxin" in prot.keywords:
                    tags[prot.accessions[0]]="Toxin"
                elif len(its := keywords.intersection(set(prot.keywords)))>0:
                    tags[prot.accessions[0]]=list(its)[0]
                else:
                    tags[prot.accessions[0]] = "Other"
    if type(model) == str:
        mod = models.Doc2Vec.load(model)
    else:
        mod = model
    vectorized = [[], []]
    for prot in tqdm(corpus):
        vector = mod.infer_vector(prot.words)
        vectorized[0].append(vector)
        if "|" in prot.tags[0]:
            acc=prot.tags[0].split("|")[1]
        else:
            acc=prot.tags[0]
        if acc in tags.keys():
            vectorized[1].append(tags[acc])
        else:
            vectorized[1].append("Other")
    return vectorized"""