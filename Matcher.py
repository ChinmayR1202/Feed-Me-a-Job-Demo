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
        
"Converts the resume into text format"        
def ExtractText(pdf):
    with pdfplumber.open(pdf) as pdf:
        first_page = pdf.pages[0]
        text = first_page.extract_text(x_tolerance=3, y_tolerance=3)

    return text

"Extracts the job descriptions from JSON files and places it all in an array"
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
        
    return descrip # Returns the array containing all the job descriptions

def ExtractInitialMatches(experience, spacyModel, encodingModel, ExpDirIndex, ExpIndirIndex, SkillsIndex, NoOfRes):
    
    ExpDirindex = faiss.read_index(ExpDirIndex) # Loads the index containing all the EXPDIR entities
    ExpIndirindex = faiss.read_index(ExpIndirIndex) # Loads the index containing all the EXPINDIR entities
    Skillsindex = faiss.read_index(SkillsIndex) # Loads the index containing all the SKILLS entities
    
    nlp = spacy.load(spacyModel) # Loads the custom spacy NER model
    queryEnts = nlp(experience) # Runs the model on the resume to extract EXPDIR, EXPINDIR and SKILLS entities
    
    model =  SentenceTransformer(encodingModel) # Loads the encoding models from sentence-transformers

    xqDir = np.array([]) # Initializes the array for hold the EXPDIR entities to query
    distDir = np.array([]) # Initializes the array to hold the distances of the queried entities
    idnDir = np.array([]) # Initializes the array to hold the IDs of the queried entities

    xqIndir = np.array([]) # Initializes the array for hold the EXPINDIR entities to query
    distIndir = np.array([]) # Initializes the array to hold the distances of the queried entities
    idnIndir = np.array([]) # Initializes the array to hold the IDs of the queried entities
    
    xqSkills = np.array([]) # Initializes the array for hold the SKILLS entities to query
    distSkills = np.array([]) # Initializes the array to hold the distances of the queried entities
    idnSkills = np.array([]) # Initializes the array to hold the IDs of the queried entities
    
    "Extracts the entities and puts them in the respective arrays"
    for token in queryEnts.ents:
        if token.label_ == "EXPdir":
            xqDir = np.append(xqDir, token.text.lower())
        elif token.label_ == "EXPindir":
            xqIndir = np.append(xqIndir, token.text.lower())
        elif token.label_ == "SKILLS":
            xqSkills = np.append(xqSkills, token.text.lower())
    
    "Removes duplicate entities"        
    xqDir = np.unique(xqDir)
    xqIndir = np.unique(xqIndir)
    xqSkills = np.unique(xqSkills)


    "Queries the entities in the FAISS vector database and stores their distances and IDs in their respective arrays"
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
                     
    "Combines the results of the EXPDIR and SKILLS entities and reshapes the array"    
    DISTDir = distDir.reshape(((len(np.unique(xqDir))) + (len(np.unique(xqSkills)))), NoOfRes) 
    IDNDir = idnDir.reshape(((len(np.unique(xqDir))) + (len(np.unique(xqSkills)))), NoOfRes)

    DISTIndir = distIndir.reshape(len(np.unique(xqIndir)), NoOfRes)
    IDNIndir = idnIndir.reshape(len(np.unique(xqIndir)), NoOfRes)
    
    "Converts the IDs to a format 'Job Description database row number'.'Number of the entity in that job description',"
    "So that every single entity gotten as a result has a unique ID number"
    "And can be linked back to its original job description"
    "Lastly removes the decimal portion as only the job description origin is needed"
    IDNDir = IDNDir/100000
    IDNDir = IDNDir.astype(int)
    
    IDNIndir = IDNIndir/100000
    IDNIndir = IDNIndir.astype(int)
    
    return DISTDir, IDNDir, DISTIndir, IDNIndir, xqDir, xqIndir, xqSkills
                
                
def SortResults(DISTDir, IDNDir, DISTIndir, IDNIndir, NoOfRes):
      
    if IDNDir.size > 0: # Sorts if any entities were found in the first place
        
        distIndex = np.argwhere(DISTDir > 0.1) # Finds entities that have a further than a distance of 0.1 (Further means less relevant)
        for i in range(len(distIndex)):
            IDNDir[tuple(distIndex[i].tolist())] = -1 # Sets those distances to -1 to not ruin the shape of the array and disregard those entities
        
        uniqueIdDir, countsDir = np.unique(IDNDir, return_counts=True) # Counts how many times a job description has appeard in the results
        uniqueIdDir = uniqueIdDir[1:] # Takes gets rid of the job descriptions with the ID -1
        countsDir = countsDir[1:] # Takes gets rid of the job descriptions with the ID -1
        countsDir = np.float64(countsDir)
        
        distIndexIndir = np.argwhere(DISTIndir > 0.1) # Does the exact same for the EXPINDIR entities
        for i in range(len(distIndexIndir)):
            IDNIndir[tuple(distIndexIndir[i].tolist())] = -1
        
        a = np.argsort(-countsDir) # Sorts the indexes of the counts array in descending order of their respective count number
        finalIndexArrangement = uniqueIdDir[a] # Sorts the ID array to the arrangement of the descending count number to get the IDs of the job descriptions with the most mentions in the query results
        finalIndexArrangement2 = finalIndexArrangement[:NoOfRes] # Trims the ID array to the number of results needed
        finalCountArrangement = countsDir[a]
        finalCountArrangement2 = finalCountArrangement[:NoOfRes]
  
    
    return finalIndexArrangement2, finalCountArrangement2, IDNDir, IDNIndir # Returns the array of the row numbers of the job descriptions that need to be shown as a result


def findMatchedSkills(finalIndexArrangement, xqDir, xqSkills, xqIndir, IDNDir, IDNIndir, descrip):
    
    xqDir = np.append(xqDir, xqSkills) # loads the entities onto a single array
    
    "Loads the job description row number, and the count of its total SKILLS entities onto respective arrays"
    df = pd.read_csv("Skills_Entitity_Count_CB.csv")
    SkillID = df['ID'].to_numpy()
    SkillCount = df['Skills'].to_numpy()
    
    "Loads the job description row number, and the count of its total EXPDIR entities onto respective arrays"
    df = pd.read_csv("ExpDir_Entitity_Count_CB.csv")
    dirID = df['ID'].to_numpy()
    dirCount = df['ExpDirEnt'].to_numpy()
    
    "Loads the job description row number, and the count of its total EXPINDIR entities onto respective arrays"
    df = pd.read_csv("ExpIndir_Entitity_Count_CB.csv")
    indirID = df['ID'].to_numpy()
    indirCount = df['ExpIndirEnt'].to_numpy()
        
    dirID = np.concatenate([dirID,SkillID]) # Concatenates the EXPDIR and SKILLS arrays
    dirCount = np.concatenate([dirCount,SkillCount]) # Concatenates the EXPDIR and SKILLS count arrays
    
    matchedSkills = []
    missingSkills = []
    matchedAction = []
    missingAction = []
    
    "Finds matched and missing skills (EXPDIR + SKILLS) and action words (EXPINDIR)"
    for index in finalIndexArrangement:
        
        jDDir = np.array([]) # Initializes the arrays
        jDIndir = np.array([]) # Initializes the arrays
        
        a = np.asarray(np.where(IDNDir == index)).tolist() # Finds the indexes in the IDNDir array where the current job description's row number appears
        skills = xqDir[a[0]].tolist() # Gets the matched skills (EXPDIR + SKILLS) entities 
        
        b = np.asarray(np.where(IDNIndir == index)).tolist() # Does the same for action words (EXPINDIR) entities
        action = xqIndir[b[0]].tolist()
        
        c = np.where(dirID == index) # Searches the skills database for all the skills that are in the current job description and gets the indices
        d = np.where(indirID == index) # searches the action words database for all the action words that are in the current job description and gets their indices
        
        allDir = dirCount[c] # Gets the respective skills
        allIndir = indirCount[d] # Gets the respective action words
        
        "If the skills or action word is not matched, that entity is a missing entitiy and puts it in is respective array"
        for i in allDir:
            if i not in skills:
                jDDir = np.append(jDDir, i)

        
        for j in allIndir:
            if j not in action:
                jDIndir = np.append(jDIndir, j)
                
        # print(allDir)
        # print(jDDir)
        
        "Joins the entities into a single string for each job description"
        matchedSkills.append(', '.join(skills))
        matchedAction.append(', '.join(action))
        missingSkills.append(', '.join(jDDir))
        missingAction.append(', '.join(jDIndir))
    
    return matchedSkills, missingSkills, matchedAction, missingAction

def MatchingScore(finalArrangement, finalCountArrangement, ExpDirEntFile, ExpIndirEntFile, SkillsEntFile, isPercent):
    
    if isPercent: # Does this if user wants a match percentage and not matched number/count
        
        "Loads the job description row number, and the count of its total SKILLS entities onto respective arrays"
        df = pd.read_csv(ExpDirEntFile)
        dirID = df['ID'].to_numpy()
        dirCount = df['Count'].to_numpy()
        
        "Loads the job description row number, and the count of its total EXPDIR entities onto respective arrays"
        df = pd.read_csv(ExpIndirEntFile)
        indirID = df['ID'].to_numpy()
        indirCount = df['Count'].to_numpy()
        
        "Loads the job description row number, and the count of its total EXPINDIR entities onto respective arrays"
        df = pd.read_csv(SkillsEntFile)
        skillsID = df['ID'].to_numpy()
        skillsCount = df['Count'].to_numpy()
    
    matchingScore = []
    
    "Gets either the match percentage or amount of skills matched"
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

    "Exports the results into an array with the row number, score and matched entities"
    result = []
    score = 0
    for index in finalArrangement:
        result.append([index, matchingScore[score], matchedSkills[score], missingSkills[score], matchedAction[score], missingAction[score]])
        score += 1
           
    return result