import pandas as pd
# AA seqs of 4 proteins
avGFP = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
amacGFP = "MSKGEELFTGIVPVLIELDGDVHGHKFSVRGEGEGDADYGKLEIKFICTTGKLPVPWPTLVTTLSYGILCFARYPEHMKMNFKSAMPEGYIQERTIFFQDDGKYKTRGEVKFEGDTLVNRIELKGMKEDGNILGHKLEYNFNSHNVYIMPDKANNGLKVNFKIRHNIEGGGVQLADHYQTNVPLGDGPVLIPINHYLSCQTAISKDRNETRDHMVFLEFFSACGHTHGMDELYK"
cgreGFP = "MTALTEGAKLFEKEIPYITELEGDVEGMKFIIKGEGTGDATTGTIKAKYICTTGDLPVPWATILSSLSYGVFCFAKYPRHIAFKSTQPDGYSQDRIISFDNDGQYDVKAKVTYENGTLYNRVTVKGTGFKSNGNILGMRVLYHSPPHAVYILPDRKNGGMKIEYNKAFDVMGGGHQMARHAQFNKPLGAWEEDYPLYHHLTVWTSFGKDPDDDETDHLTIVEVIKAVDLETYR"
ppluGFP2 = "MPAMKIECRITGTLNGVEFELVGGGEGTPEQGRMTNKMKSTKGALTFSPYLLSHVMGYGFYHFGTYPSGYENPFLHAINNGGYTNTRIEKYEDGGVLHVSFSYRYEAGRVIGKVVGTGFPEDSVIFTDKIIRSNATVEHLHPMGDNVLVGSFARTFSLRDGGYYSFVVDSHMHFKSAIHPSILQNGGPMFAFRRVEELHSNTELGIVEYQHAFKTPIAFA"
# 以上是原始数据，读变体：
raw_data = pd.read_csv('data.csv')
# print(raw_data)
# print(raw_data.shape)
mut_sequences = []
for index, data in raw_data.iterrows():
    seq = data[0]
    wd_type = data[1]
    # print(seq, wd_type)

    if wd_type == "avGFP":
        wd_type_seq = avGFP
    elif wd_type == "amacGFP":
        wd_type_seq = amacGFP
    elif wd_type == "cgreGFP":
        wd_type_seq = cgreGFP
    elif wd_type == "ppluGFP":
        wd_type_seq = ppluGFP2
    else: 
        print("Error: no such GFP type")
        break
    mutations = seq.split(":")

    mut_seq = wd_type_seq
    
    for mutation in mutations:
        if mutation == "WT":
            mut_seq = wd_type_seq
        else:
            pos = int(mutation[1:-1])
            aa = mutation[-1]
            mut_seq = mut_seq[:pos] + aa + mut_seq[pos+1:]
        data["mut_seq"] = mut_seq
    
    mut_sequences.append(mut_seq)
print(len(mut_sequences))
# 将突变后的序列添加到原始数据框
raw_data['mut_seq'] = mut_sequences

# 保存到新的CSV文件
raw_data.to_csv('data_with_mut_seq.csv', index=False)




