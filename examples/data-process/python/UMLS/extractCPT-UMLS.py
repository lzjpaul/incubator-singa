"""
Extract the CPT/HCPCS hierarchy from UMLS file.

"""
import csv
import re
from collections import OrderedDict
import bisect
import json


def get_bounds(code):
    grp = code.split(":")
    grp = grp[1].replace(" ", "").split("-")
    lb = grp[0]
    ub = grp[1]
    return lb, ub


def get_raw_code(umlsFile, keyWord):
    codeDict = {}
    for row in csv.reader(open(umlsFile, "rb"), delimiter="|"):
        if row[11] == keyWord and row[12] == "PT":
            codeDict[row[13]] = row[14]
    return codeDict


def get_hierarchy(umlsFile, keyword, levelKeyword):
    catHier = {}
    for row in csv.reader(open(umlsFile, "rb"), delimiter="|"):
        if not row[11] == keyword:
            continue
        # otherwise it is so do something
        if re.search(levelKeyword, row[13]):
            lb, ub = get_bounds(row[13])
            catHier[lb] = {"upper": ub, "desc": row[14]}
    return OrderedDict(sorted(catHier.items()))


def find_value(kDict, k):
    ind = bisect.bisect_right(kDict.keys(), k)
    return (ind - 1)


def lookup_item(code, levelDict):
    level1 = find_value(levelDict[0], code)
    level2 = find_value(levelDict[1], code)
    level3 = find_value(levelDict[2], code)
    return {"L1": level1, "L2": level2, "L3": level3}


def find_code_levels(codeDict, levelDictList, codeType, codeLevels={}):
    for k, v in codeDict.items():
        codeLevels[k] = lookup_item(k, levelDictList)
        codeLevels[k]['desc'] = v
        codeLevels[k]['type'] = codeType
    return codeLevels


def gen_level_desc(levelDict):
    return [v['desc'] for v in levelDict.itervalues()]


def build_code_dict(umlsFile, rawKeyword, hierKeyword):
    codeDict = get_raw_code(umlsFile, rawKeyword)
    codeHier1 = get_hierarchy(umlsFile, hierKeyword, "Level 1")
    codeHier2 = get_hierarchy(umlsFile, hierKeyword, "Level 2")
    codeHier3 = get_hierarchy(umlsFile, hierKeyword, "Level 3")
    hierLevels = [codeHier1, codeHier2, codeHier3]
    codeLevels = find_code_levels(codeDict, hierLevels, rawKeyword)
    codeHierDesc = {}
    codeHierDesc["L1"] = gen_level_desc(codeHier1)
    codeHierDesc["L2"] = gen_level_desc(codeHier2)
    codeHierDesc["L3"] = gen_level_desc(codeHier3)
    return codeLevels, hierLevels, codeHierDesc


def append_code_dict(umlsFile, rawKeyword, codeLevels, codeHier):
    addCodeDict = get_raw_code(umlsFile, rawKeyword)
    for k, v in addCodeDict.items():
        if k in codeLevels:
            continue
        codeLevels[k] = lookup_item(k, codeHier)
        codeLevels[k]['desc'] = v
        codeLevels[k]['type'] = rawKeyword
        print codeLevels[k]
    return codeLevels


def impute_unknown(codeLevels, cptHL, hcpcsHL):
    procFull = json.load(open("data/hcpc-dict.json", "rb"))
    for k, v in procFull.iteritems():
        if k in codeLevels:
            continue
        if k[0].isdigit():
            codeLevels[k] = lookup_item(k, cptHL)
            codeLevels[k]['desc'] = v
            codeLevels[k]['type'] = 'CPT'
            continue
        codeLevels[k] = lookup_item(k, hcpcsHL)
        codeLevels[k]['desc'] = v
        codeLevels[k]['type'] = 'HCPCS'
    return codeLevels


def main():
    recentFile = "/data/zhaojing/MRCONSO.RRF"
    cptCodes, cptHL, cptHDesc = build_code_dict(recentFile, "CPT", "MTHCH")
    hcpcsCodes, hcpcsHL, hcpcsHDesc = build_code_dict(recentFile, "HCPCS",
                                                      "MTHHH")
    homeDir = "/data/zhaojing/"
    prevUMLSFiles = [homeDir + "MRCONSO.RRF"
         #            homeDir + "umls-2009/2009AB/META/MRCONSO.RRF",
          #           homeDir + "umls-2008/2008AB/META/MRCONSO.RRF",
           #          homeDir + "umls-2007/2007AC/META/MRCONSO.RRF",
            #         homeDir + "umls-2005/2005AC/META/MRCONSO.RRF"
                     ]
    for umlsFile in prevUMLSFiles:
        cptCodes = append_code_dict(umlsFile, "CPT", cptCodes, cptHL)
        hcpcsCodes = append_code_dict(umlsFile, "HCPCS", hcpcsCodes, hcpcsHL)
    procDict = OrderedDict(sorted(cptCodes.items() + hcpcsCodes.items(),
                                  key=lambda t: t[0]))
    #procDict = impute_unknown(procDict, cptHL, hcpcsHL)
    with open('/data/zhaojing/proc2Levels.json', 'w') as outfile:
        json.dump(procDict, outfile, indent=2)
    procLevels = {"CPT": cptHDesc, "HCPCS": hcpcsHDesc}
    with open('/data/zhaojing/procLevelLabels.json', 'w') as outfile:
        json.dump(procLevels, outfile, indent=2)
    ## process the dictionary
    lookupDict = {k: v['type'] + " " + str(v['L2'])
                  for k, v in procDict.items()}
    with open('/data/zhaojing/proc2UMLS.json', 'w') as outfile:
        json.dump(lookupDict, outfile, indent=2)
    return

if __name__ == "__main__":
    main()

#missing codes - vandy
# set(['D0321', '35546', 'J7320', '99142', 'J9306', '99141', 'D4355',
# 'D8670.20', 'G0338', 'D5110', 'Q2051', '90724', '64443', '99173.5',
# 'D9110', '33253', 'D2335', 'Q0086', 'D6010', 'D2331', 'D2330',
# 'D2332', 'D0350', '88151', 'D7880', '11050', '79030', '76934',
# 'D7950', 'D3120', '85095', 'D7960', '98960.2', '93536', '99999',
# 'TAX01', 'D2150', 'C99999', '61855', 'OPH16', 'D5213', 'D5212',
# 'D2931', 'C99999.2', 'D5214', 'D8680', '76062', '17310', 'Q4095',
# '19220', 'D5510', 'Q4092', '11731', '83716', '97703', 'Q4098',
# 'D8080', 'C9020', '70541', '80183', 'C8954', 'C8955', 'Q3021',
# 'C8950', 'C8951', 'C8952', 'C8953', 'D5520', '90799', 'D1351',
# '85021', '85022', '85023', '85024', '80054', '85029', '80058',
# '00049', 'REF03', 'REF01', 'REF04', 'REF05', '00041', '00040',
# '00043', '00045', '00044', '00047', '00046', '78810', '19182',
# '19180', 'Q2005', 'Q2003', 'GO283', 'OB999', '36488', '56308',
# '36534', '36535', 'D4999', '36537', 'D3310', '36533', '94651',
# '94650', '92981.1', '94657', '94656', '92599', '56304', 'J3489',
# '90471.5', 'D7510', '56301', 'C1144', '56300', '01999.8', '01999.1',
# '01999.2', '01999.5', '01999.4', '01999.7', '01999.6', '64830',
# '76938', '86313', '25620', 'D2752', 'D2750', '56309', '19240',
# 'D7250', '56305', '76012', '56307', '56306', 'G0107', 'C9007',
# '56303', '56302', 'D0220', '79000', '76355', 'C9287', '76490',
# 'D5411', 'D5410', '92928.1', 'Q4053', 'Q4084', '29815', 'G0033',
# 'Q4055', 'Q0124', 'G0217', 'G0215', '78715', 'C9261', 'G0212',
# 'G0211', 'C9105', 'G0063', 'C9103', 'IMRT1', 'G0195', 'G0196',
# 'C8916', '56399', '17101', 'D9940', 'D9230', 'C9210', 'C9212',
# 'C9214', '83720', 'AG000', '15350', 'EOT00', 'C9280', '17002',
# 'D7140', 'D2999', '66984LT', '00016', '00015', '00012', '90649.1',
# '00011', '00018', '00019', '09650', '66250.1', 'C1775', 'C1774',
# '84903', 'Q4052', 'D4342', 'D4341', 'C9999', '80006', 'E0754',
# 'Q2046', '90730', '92524', '44200', 'D7240', 'D7241', 'J7051',
# '19200', '15625', '22150', 'D0340', 'Q3002', 'J3395', 'D7953',
# '33249.1', 'D9630', 'C1207', '96410', 'DEN999', '96412', 'C9229',
# 'Q0187', 'D2920', '92980.1', '9136', 'D59992', 'D2140', 'C9227',
# 'D2393', 'D2392', 'D2391', 'ANES888', 'D2399', '75998', 'D6545',
# 'D8691', 'D8692', '91032', '91033', '56341', '56340', 'Q4083',
# '17306', '17307', '17304', '17305', 'C3025.23', 'D8090', 'J2352',
# 'C3025.29', '90633.1', 'C1305', '15001', '15000', 'Q3030', 'G0242',
# 'C1094', '00054', 'C1093', 'C1091', 'C1098', 'D4910', '97601',
# '00052', '00053', '43639', '00051', '00056', 'D8999', '36493',
# '36491', 'C2632', 'D0160', '43638', '45378.1', 'D7110', 'D9910',
# 'Q9943', 'S2363', 'S2362', '92211', '19083', '19081', '88180',
# '19084', '86588', 'D3960', 'A9520', 'D5211', '0083T', 'D6059',
# '86064', 'J0151', 'D8670.9', 'D5120', '76365', '56317', '56314',
# '56315', 'D2740', '76360', '76362', 'HDR01', 'D6999', 'D0230',
# 'G0214', 'D7230', 'Q0136', 'Q0137', 'A6364', '88171', '88170',
# 'G0025', 'ANES000', 'D2130', '78727', '78726', '99313', '99312',
# '99311', '79900', '90686', 'D2910', 'D9951', '00099', 'D0431',
# '76005', 'D7471', '76003', 'C1067', 'G0296', 'D0120', 'C9000',
# 'C9009', 'C9008', 'Q3005', 'G0040', '15351', '93738', '93737',
# '99262', '99261', 'Q3009', 'Q3008', '45378.2', 'D2980', 'D9450',
# '0328T', '92523', '00021', '00020', '00026', '00025', '00029',
# '00028', 'D3430', 'J7188', 'D8670', 'G0218', '76986', 'D1110',
# 'CPK05', 'CPK01', '93607', 'A62230', '90472.5', 'J7508', '33245',
# '83715', '0279T', '96530', 'V2741', 'D7270', 'A9515', 'A9514',
# 'D8670.3', 'D0272', 'D0270', 'A9519', 'D0274', '19281', 'G0253',
# 'D8670.1', '19285', 'D2799', 'D8670.7', '77420', 'D2792', 'D2790',
# '15831', '16015', 'D8670.4', '96408', '37720', 'D8670.5', '96400',
# 'T90784', 'D2950', 'D2954', '78704', 'G0125', 'D1201', 'C9230',
# 'D1203', 'D1204', 'D1205', 'D1206', 'D0470', 'D0471', 'D2116',
# 'D5820', 'D2385', 'D2386', 'D9999', '67350', '15580', '92551.5',
# 'D8660', '56353', '56350', '56351', '56356', '56354', '76040',
# 'C9126', 'C3025.12', 'D8670.19', 'C9120', 'C3025.11', '97504',
# 'D8670.15', 'D8670.16', 'D8670.17', 'D8670.10', 'D8670.11',
# 'D8670.12', 'D8670.13', '96372.1', 'G0230', '80175', '80177', '00060',
# 'D9310', '85595', 'D2962', '31250', '80092', '36489', '80091',
# 'G0347', 'C9240', 'D0150', 'G0348', 'REF02', 'D7120', 'MKT04',
# '80009', 'G0224', 'MKT03', 'MKT02', 'GOOO9', '80004', '80007',
# 'G0225', 'MKT08', 'D0418', 'J7318', 'D0330', 'J7316', 'J7317',
# 'C17380.27', 'C1202', 'C1201', 'C1200', 'D2960', 'D1120', '53670',
# '89360', '86379', '00007', 'Q2024', 'D7321', 'Q2021', 'D6241',
# 'D6056', 'J0880', 'EB2C1', 'D5130', 'D7220', 'D6931', '55859',
# '76375', '76370', 'C3025.1', 'D0210', 'C3025.2', 'INF04', 'INF05',
# 'INF01', 'INF02', 'INF03', '76778', '79020', 'D3410', 'C9246',
# 'D5610', 'D1330', '99301', '99302', '99303', '31260', 'D7310',
# 'A4646', 'A4647', 'A4644', 'A4645', 'D7997', '61862', '80008',
# 'J1563', 'D5810', 'D6750', 'J2000', '76075', '76076', '29909',
# '99274', '99275', 'G0463', '99271', '99272', '99273', 'C9223',
# 'Y1405', '15342', '15343', 'G0262', 'Q3010', 'D4263', '82130',
# '90788', '80049', '90782', '90781', '90780', '90784', '56316',
# '00038', '64441', '00036', '00030', '00031', '00032', '00033',
# 'Q2019', 'Q2044', 'Q2014', '56313', 'D0180', 'A9603', '19160',
# '19162', '93757', '85102', '92589', '36520', '90659', '99199.1',
# 'D0240', 'G0351', '99025', 'J2996', 'C9010', '77430', 'J3245',
# '96100', 'T2023U2', '99263', 'D2940', '52340', '76020', 'D5999',
# 'OPH21', 'D7210', '62275', '87178', '87072', 'J1442', '62278',
# '62279', '52338', 'D8670.8', 'G0001', 'G0006', 'V2740', 'G0005',
# 'D8670.2', '96115', '96117', 'D8670.6', '52335', '52336', '52337',
# '17100', 'C9112', '17102', '76560', 'PSI01', '17105', 'C9115',
# 'C9119', 'K0549', '76093', '76092', '76091', '76090', '76096',
# '76095', '76094', 'G0222', 'D8670.18', 'G0220', 'G0221', 'G0227',
# '92012.1', '49085', '92014.1', '92014.2', 'C3025.4', 'D9240', 'D9241',
# 'D9242', 'C3025.3', 'D2160', 'D8670.14', 'C9205', 'C9204', 'C9202',
# 'D9972', 'D9971', 'D0140', 'T90672', '00002', '92004.2', '94665',
# '92004.1', 'EB3D1', '64644', 'D9430', '80012', 'D7130', 'D6240',
# '00005', '00004', 'V0799', '00006', '00001', '00003', '80019',
# '80016', '00009', '00008'])
