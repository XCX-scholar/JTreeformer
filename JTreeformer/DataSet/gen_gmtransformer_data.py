folder="path/to/dataset"
'''vocab for SMILES'''
d=['I', '5', 'N', '1', '@', '+', '=', '#', 'O', '\\', '6', 'S', 'o', 'B', 's', '2', '(', '3', 'H', 'c', '4', '-', 'l', ')', 'Br', 'C', 'n', '[', 'F', '/', ']', 'P','[n' ,'[nH' ,'[nH]'  ,'[N' ,'[N+' , '[N+]' ,'Br','[O','[O-','[O-]','Sn']
d=set(d)

def smiles2stentance(smiles:str,d:set):
    cur_pos=0
    size=len(smiles)
    sentance=""
    while(cur_pos<size):
        end_pos = cur_pos + 1
        if cur_pos==size-1:
            sentance=sentance+smiles[cur_pos]+"\n"
            break
        else:
            stop=False
            while(not stop):
                if smiles[cur_pos:end_pos] in d:
                    end_pos+=1
                else:
                    stop=True
                if end_pos==size+1:
                    stop=True
            end_pos-=1
            sentance=sentance+smiles[cur_pos:end_pos]
            if end_pos==size:
                sentance=sentance+"\n"
            else:
                sentance=sentance+" "
        # print(cur_pos,end_pos)
        cur_pos=end_pos
        # print(sentance)
    return sentance

with open(folder +"zinc_train.smi", 'r') as file:
    smiles = [x.strip() for x in file.readlines()]

# print(smiles2stentance(smiles[0],d))
#
# tgt_file=open("D:\\mol_project\\GMTransformer-main\\GMTransformer\\"+"zinc_atom_valid.txt", 'w')
# for i,s in enumerate(smiles):
#     # print(i)
#     tgt_file.write(smiles2stentance(s,d))
#     if i%5000==0:
#         print(i)
# tgt_file.close()
num_batch={i:0 for i in range(96)}
for s in smiles:
    num_batch[len(s)]+=1

total=0
for k,v in num_batch.items():
    total+=k*v
print(total/10000)

# batch_num:1153