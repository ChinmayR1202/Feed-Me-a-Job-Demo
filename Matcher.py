import sys
sys.path.append("/mnt/efs/matcher-depend")
import spacy
import pdfplumber
import pandas as pd
import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import csv
import os
os.chdir("/mnt/efs/src")
        
class Match:
    def __init__(self, resume, isPercent, ExpDirindex, ExpIndirindex, Skillsindex, NoOfRes):

        self.resume = resume
        
        self.ExpDirindex = ExpDirindex
        self.ExpIndirindex = ExpIndirindex
        self.Skillsindex = Skillsindex
        self.NoOfRes = NoOfRes
        
        self.isPercent = isPercent
        
        self.experience = ExtractText(self.resume)
        self.descriptions = ExtractJobDescriptions()
        
        self.DISTDir, self.IDNDir, self.DISTIndir, self.IDNIndir, self.xqDir, self.xqIndir, self.xqSkills = ExtractInitialMatches(self.experience, "en_model", "all-distilroberta-v1", self.ExpDirindex, self.ExpIndirindex, self.Skillsindex, self.NoOfRes)        
        self.FinalArrangement, self.FinalCountArrangement, self.IDNDir, self.IDNIndir = SortResults(self.DISTDir, self.IDNDir, self.DISTIndir, self.IDNIndir, self.NoOfRes)         
        self.matchedSkills, self.missingSkills, self.matchedAction, self.missingAction = findMatchedSkills(self.FinalArrangement, self.xqDir, self.xqSkills, self.xqIndir, self.IDNDir, self.IDNIndir, self.descriptions)
        self.MatchingScore = MatchingScore(self.FinalArrangement, self.FinalCountArrangement,  "dirCount.csv", "indirCount.csv", "skillsCount.csv", self.isPercent)
        self.Results = Results(self.FinalArrangement, self.MatchingScore, self.matchedSkills, self.missingSkills, self.matchedAction, self.missingAction)
        
        
def ExtractText(pdf):
    with pdfplumber.open(pdf) as pdf:
        first_page = pdf.pages[0]
        text = first_page.extract_text(x_tolerance=3, y_tolerance=3)

    return text

def ExtractJobDescriptions(jobDescriptions=None):
    descrip = []
    with open('CareerBuilder_job_details_data_1.json', 'r', encoding="utf-8") as openfile:
        JD_raw = openfile.read()
        JD_list = json.loads(JD_raw)
    for i in range(len(JD_list)):
        descrip.append([JD_list[str(i)].get('joblink'), JD_list[str(i)].get('job_description')])
        
    with open('CareerBuilder_job_details_data_2.json', 'r', encoding="utf-8") as openfile:
        JD_raw = openfile.read()
        JD_list = json.loads(JD_raw)
    for i in range(len(JD_list)):
        descrip.append([JD_list[str(i)].get('joblink'), JD_list[str(i)].get('job_description')])
    
    with open('CareerBuilder_job_details_data_3.json', 'r', encoding="utf-8") as openfile:
        JD_raw = openfile.read()
        JD_list = json.loads(JD_raw)
    for i in range(len(JD_list)):
        descrip.append([JD_list[str(i)].get('joblink'), JD_list[str(i)].get('job_description')])
    
    with open('CareerBuilder_job_details_data_4.json', 'r', encoding="utf-8") as openfile:
        JD_raw = openfile.read()
        JD_list = json.loads(JD_raw)
    for i in range(len(JD_list)):
        descrip.append([JD_list[str(i)].get('joblink'), JD_list[str(i)].get('job_description')])
        
    with open('CareerBuilder_job_details_data_5.json', 'r', encoding="utf-8") as openfile:
        JD_raw = openfile.read()
        JD_list = json.loads(JD_raw)
    for i in range(len(JD_list)):
        descrip.append([JD_list[str(i)].get('joblink'), JD_list[str(i)].get('job_description')])
        
    return descrip

def ExtractInitialMatches(experience, spacyModel, encodingModel, ExpDirIndex, ExpIndirIndex, SkillsIndex, NoOfRes):
    
    ExpDirindex = faiss.read_index(ExpDirIndex)
    ExpIndirindex = faiss.read_index(ExpIndirIndex)
    Skillsindex = faiss.read_index(SkillsIndex)
    
    nlp = spacy.load(spacyModel)
    queryEnts = nlp(experience)
    
    model =  SentenceTransformer(encodingModel)

    xqDir = np.array([])
    distDir = np.array([])
    idnDir = np.array([])

    xqIndir = np.array([])
    distIndir = np.array([])
    idnIndir = np.array([])
    
    xqSkills = np.array([])
    distSkills = np.array([])
    idnSkills = np.array([])
    
    for token in queryEnts.ents:
        if token.label_ == "EXPdir":
            xqDir = np.append(xqDir, token.text.lower())
        elif token.label_ == "EXPindir":
            xqIndir = np.append(xqIndir, token.text.lower())
        elif token.label_ == "SKILLS":
            xqSkills = np.append(xqSkills, token.text.lower())
            
    xqDir = np.unique(xqDir)
    xqIndir = np.unique(xqIndir)
    xqSkills = np.unique(xqSkills)

    for token in xqDir:
        query = model.encode([token])
        d, i = ExpDirindex.search(query, NoOfRes)
        distDir = np.append(distDir, d)
        idnDir = np.append(idnDir, i)            

           
    for token in xqIndir:
        query = model.encode([token])
        d, i = ExpIndirindex.search(query, NoOfRes)
        distIndir = np.append(distIndir, d)
        idnIndir = np.append(idnIndir, i)
        
        
    for token in xqSkills:
        query = model.encode([token])
        d, i = Skillsindex.search(query, NoOfRes)
        distDir = np.append(distDir, d)
        idnDir = np.append(idnDir, i)
                     
    DISTDir = distDir.reshape(((len(np.unique(xqDir))) + (len(np.unique(xqSkills)))), NoOfRes) 
    IDNDir = idnDir.reshape(((len(np.unique(xqDir))) + (len(np.unique(xqSkills)))), NoOfRes)

    DISTIndir = distIndir.reshape(len(np.unique(xqIndir)), NoOfRes)
    IDNIndir = idnIndir.reshape(len(np.unique(xqIndir)), NoOfRes)
    
    IDNDir = IDNDir/100000
    IDNDir = IDNDir.astype(int)
    
    IDNIndir = IDNIndir/100000
    IDNIndir = IDNIndir.astype(int)
    
    return DISTDir, IDNDir, DISTIndir, IDNIndir, xqDir, xqIndir, xqSkills
                
                
def SortResults(DISTDir, IDNDir, DISTIndir, IDNIndir, NoOfRes):
      
    if IDNDir.size > 0:
        
        distIndex = np.argwhere(DISTDir > 0.1)
        for i in range(len(distIndex)):
            IDNDir[tuple(distIndex[i].tolist())] = -1
        
        uniqueIdDir, countsDir = np.unique(IDNDir, return_counts=True)
        uniqueIdDir = uniqueIdDir[1:]
        countsDir = countsDir[1:]
        countsDir = np.float64(countsDir)
        
        distIndexIndir = np.argwhere(DISTIndir > 0.1)
        for i in range(len(distIndexIndir)):
            IDNIndir[tuple(distIndexIndir[i].tolist())] = -1
        
        a = np.argsort(-countsDir)
        finalIndexArrangement = uniqueIdDir[a]
        finalIndexArrangement2 = finalIndexArrangement[:NoOfRes]
        finalCountArrangement = countsDir[a]
        finalCountArrangement2 = finalCountArrangement[:NoOfRes]
  
    
    return finalIndexArrangement2, finalCountArrangement2, IDNDir, IDNIndir


def findMatchedSkills(finalIndexArrangement, xqDir, xqSkills, xqIndir, IDNDir, IDNIndir, descrip):
    
    xqDir = np.append(xqDir, xqSkills)
    # nlp = spacy.load("en_model")
    
    df = pd.read_csv("Skills_Entitity_Count_CB.csv")
    SkillID = df['ID'].to_numpy()
    SkillCount = df['Skills'].to_numpy()
    
    df = pd.read_csv("ExpDir_Entitity_Count_CB.csv")
    dirID = df['ID'].to_numpy()
    dirCount = df['ExpDirEnt'].to_numpy()
    
    df = pd.read_csv("ExpIndir_Entitity_Count_CB.csv")
    indirID = df['ID'].to_numpy()
    indirCount = df['ExpIndirEnt'].to_numpy()
        
    dirID = np.concatenate([dirID,SkillID])
    dirCount = np.concatenate([dirCount,SkillCount])
    
    matchedSkills = []
    missingSkills = []
    matchedAction = []
    missingAction = []
    
    for index in finalIndexArrangement:
        
        jDDir = np.array([])
        jDIndir = np.array([])
        
        # jDEnts = nlp(str(descrip[index - 1]))
        
        a = np.asarray(np.where(IDNDir == index)).tolist()
        skills = xqDir[a[0]].tolist()
        
        b = np.asarray(np.where(IDNIndir == index)).tolist()
        action = xqIndir[b[0]].tolist()
        
        c = np.where(dirID == index)
        d = np.where(indirID == index)
        
        allDir = dirCount[c]
        allIndir = indirCount[d]
        
        for i in allDir:
            if i not in skills:
                jDDir = np.append(jDDir, i)

        
        for j in allIndir:
            if j not in action:
                jDIndir = np.append(jDIndir, j)
                
        # print(allDir)
        # print(jDDir)
        
        matchedSkills.append(', '.join(skills))
        matchedAction.append(', '.join(action))
        missingSkills.append(', '.join(jDDir))
        missingAction.append(', '.join(jDIndir))
    
    return matchedSkills, missingSkills, matchedAction, missingAction

def MatchingScore(finalArrangement, finalCountArrangement, ExpDirEntFile, ExpIndirEntFile, SkillsEntFile, isPercent):
    
    if isPercent: 
        df = pd.read_csv(ExpDirEntFile)
        dirID = df['ID'].to_numpy()
        dirCount = df['Count'].to_numpy()
        
        df = pd.read_csv(ExpIndirEntFile)
        indirID = df['ID'].to_numpy()
        indirCount = df['Count'].to_numpy()
        
        df = pd.read_csv(SkillsEntFile)
        skillsID = df['ID'].to_numpy()
        skillsCount = df['Count'].to_numpy()
    
    matchingScore = []
    
    i = 0
    for index in finalArrangement:
        if isPercent:
            a = np.where(skillsID == int(index))
            b = np.where(dirID == int(index))

            SkillsCount = skillsCount[a]
            DirCount = dirCount[b]

            matchPer = (finalCountArrangement[i]/(DirCount[0] + SkillsCount[0]))*100
            matchingScore.append(matchPer)
        else:
            matchingScore.append(finalCountArrangement[i])
        i += 1
        
    return matchingScore
    

def Results(finalArrangement, matchingScore, matchedSkills, missingSkills, matchedAction, missingAction):

    result = []
    score = 0
    for index in finalArrangement:
        result.append([index, matchingScore[score], matchedSkills[score], missingSkills[score], matchedAction[score], missingAction[score]])
        score += 1
           
    return result