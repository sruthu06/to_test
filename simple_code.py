from tkinter import *
import numpy as np
import pandas as pd

from code import l1,disease

l2=[]

for i in range(0,len(l1)):
    l2.append(0)



df=pd.read_csv("Prototype.csv")

#Replace the values in the imported file by pandas by the inbuilt function replace in pandas.
df.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
'Migraine':11,'Cervical spondylosis':12,
'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
'Impetigo':40}},inplace=True)

#check the df 
#print(df.head())

X= df[l1]

#print(X)

y = df[["prognosis"]]
np.ravel(y)

#print(y)

#Read a csv named Testing.csv

tr=pd.read_csv("Prototype-1.csv")

#Use replace method in pandas.

tr.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
'Migraine':11,'Cervical spondylosis':12,
'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
'Impetigo':40}},inplace=True)

X_test= tr[l1]
y_test = tr[["prognosis"]]

#print(y_test)

np.ravel(y_test)




########################################################################
########################################################################


def DecisionTree(Symptom1,Symptom2,Symptom3,Symptom4,Symptom5):

    from sklearn import tree

    clf3 = tree.DecisionTreeClassifier() 
    clf3 = clf3.fit(X,y)

    from sklearn.metrics import accuracy_score
    y_pred=clf3.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(accuracy_score(y_test, y_pred,normalize=False))
    
    
    
    
    
    psymptoms=[]
    l2=[]
    for i in range(0,len(l1)):
        l2.append(0)
    

    
    

        
#     Symptom1='skin_peeling'
#     Symptom2='silver_like_dusting'
#     Symptom3='small_dents_in_nails'
#     Symptom4='muscle_pain'
#     Symptom5='inflammatory_nails'

    psymptoms = [Symptom1,Symptom2,Symptom3,Symptom4,Symptom5]
#     print("psymptoms",psymptoms)

    for k in range(0,len(l1)):
        for z in psymptoms:
            if(z==l1[k]):
                l2[k]=1
    print("psymptoms",psymptoms)
    print("l2",l2)
    inputtest = [l2]
    predict = clf3.predict(inputtest)
    print("predict",predict)
    predicted=predict[0]

    h='no'
    for a in range(0,len(disease)):
        if(predicted == a):
            h='yes'
            break
#     print(disease[a])


    if (h=='yes'):
            print(disease[a])

    else:
        print("Not Found")




def randomforest(Symptom1,Symptom2,Symptom3,Symptom4,Symptom5):
    from sklearn.ensemble import RandomForestClassifier
    clf4 = RandomForestClassifier()
    clf4 = clf4.fit(X,np.ravel(y))

    # calculating accuracy 
    from sklearn.metrics import accuracy_score
    y_pred=clf4.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(accuracy_score(y_test, y_pred,normalize=False))
    psymptoms=[]
    l2=[]
    for i in range(0,len(l1)):
        l2.append(0)


        
#     Symptom1='skin_peeling'
#     Symptom2='silver_like_dusting'
#     Symptom3='small_dents_in_nails'
#     Symptom4='muscle_pain'
#     Symptom5='inflammatory_nails'
    
    psymptoms = [Symptom1,Symptom2,Symptom3,Symptom4,Symptom5]
    
    


    for k in range(0,len(l1)):
        for z in psymptoms:
            if(z==l1[k]):
                l2[k]=1
    print("psymptoms",psymptoms)
    print("l2",l2)
    inputtest = [l2]
    predict = clf4.predict(inputtest)
    predicted=predict[0]

    h='no'
    for a in range(0,len(disease)):
        if(predicted == a):
            h='yes'
            break

    if (h=='yes'):
        print(disease[a])
    else:
        print("Not Found")


def NaiveBayes(Symptom1,Symptom2,Symptom3,Symptom4,Symptom5):
    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    gnb=gnb.fit(X,np.ravel(y))

    from sklearn.metrics import accuracy_score
    y_pred=gnb.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(accuracy_score(y_test, y_pred,normalize=False))
    psymptoms=[]
    l2=[]
    for i in range(0,len(l1)):
        l2.append(0)

     
        
#     Symptom1='skin_peeling'
#     Symptom2='silver_like_dusting'
#     Symptom3='small_dents_in_nails'
#     Symptom4='muscle_pain'
#     Symptom5='inflammatory_nails'

    psymptoms = [Symptom1,Symptom2,Symptom3,Symptom4,Symptom5]
    for k in range(0,len(l1)):
        for z in psymptoms:
            if(z==l1[k]):
                l2[k]=1

    print("psymptoms",psymptoms)
    print("l2",l2)
    
    inputtest = [l2]
    predict = gnb.predict(inputtest)
    predicted=predict[0]

    h='no'
    for a in range(0,len(disease)):
        if(predicted == a):
            h='yes'
            break

    if (h=='yes'):
        print(disease[a])
    else:
        print("Not Found")


Symptom1='internal_itching'
Symptom2='bladder_discomfort'
Symptom3='foul_smell_of urine'
Symptom4='continuous_feel_of_urine'
Symptom5='passage_of_gases'

DecisionTree(Symptom1,Symptom2,Symptom3,Symptom4,Symptom5)
randomforest(Symptom1,Symptom2,Symptom3,Symptom4,Symptom5)
NaiveBayes(Symptom1,Symptom2,Symptom3,Symptom4,Symptom5)
